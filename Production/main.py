import functions_framework
import firebase_admin
from firebase_admin import firestore
from google.cloud import secretmanager
from google.cloud import storage
import json
import logging
import openai
import pandas as pd
import faiss
import numpy as np
from datetime import datetime, timezone
from google.cloud import translate
import uuid
import requests
from io import BytesIO
import re

logging.basicConfig(level=logging.INFO)

# Initialise Firebase if not already
if not firebase_admin._apps:
    firebase_admin.initialize_app()
db_olivia = firestore.client(database_id='olivia')

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, x-firebase-appcheck"
}
 

def access_secret(secret_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/1052202528756/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


dialog360_key = access_secret("360DIALOG_API_KEY")

# Download data from storage
data_string = storage.Client().bucket("customer-service-info").blob("olivia_information_vectors.json").download_as_string()
data = json.loads(data_string)


# ASSIGN ALL RELEVANT VALUES HERE
models = data["models"]

assistant_model = models["assistant_model"]
moderation_model = models["moderation_model"]
moderation_self_harm_categories = models["moderation_self_harm_categories"]
transcription_model = models["transcription_model"]
summary_query_model = models["summary_query_model"]
embedding_model = models['embedding_model']


# AND HERE
hyperparams = data["hyperparams"]

k = hyperparams["k"] # Number of nearest neighbours to retrieve
p_cosine_min = hyperparams["p_cosine_min"] # cosine similarity threshold
p_audio = hyperparams["p_audio"] # minimum threshold for average probability per token transcribed
p_langdetect = hyperparams["p_lang_detect"] # minimum threshold for probability of language detected
p_human_escalation = hyperparams["p_human_escalation"] # minimum threshold for cosine similarity to escalate to human support
restart_summary_every = hyperparams["restart_summary_every"] # summary rewritten from entire thread at these regular intervals


# PROMPRTS AND TEMPLATES
prompts = data["prompts"]

generate_response_prompt = prompts["generate_response_prompt"]
permitted_topics = prompts["permitted_topics"]
banned_topics = prompts["banned_topics"]
summary_rewrite_template = prompts["summary_rewrite_user_template"]
query_rewrite_system_prompt = prompts["query_rewrite_system_prompt"]
query_rewrite_user_template = prompts["query_rewrite_user_template"]


info_df = pd.DataFrame(data["information"])
human_escalation_df = pd.DataFrame(data["human_escalation"])



supported_languages = data["supported_languages"]
phone_to_language = data["phone_to_language"]


# assign all the preset responses from our downloaded data
preset_responses = data["preset_responses"]

greeting_responses = preset_responses["greeting_responses"]
gratitude_responses = preset_responses["gratitude_responses"]
resolved_responses = preset_responses["resolved_responses"]
not_understood_responses = preset_responses["not_understood_responses"]
flagged_content_responses = preset_responses["flagged_content_responses"]
self_harm_responses = preset_responses["self_harm_responses"]
unsupported_media_responses = preset_responses["unsupported_media_responses"]
hard_escalate_responses = preset_responses["hard_escalate_responses"]


greeting_triggers = data["greeting_triggers"]
gratitude_triggers = data["gratitude_triggers"]


openai_client = openai.OpenAI(api_key = access_secret("OPENAI_API_KEY"))

translate_client = translate.TranslationServiceClient()



# Create matrix of information vectors
info_matrix = np.vstack(info_df['vector_embedding'].values).astype('float32')
human_escalation_matrix = np.vstack(human_escalation_df['vector_embedding'].values).astype('float32')


# Generate FAISS index of information embeddings
info_dim = info_matrix.shape[1]
info_index = faiss.IndexFlatIP(info_dim)
info_index.add(info_matrix)

# Generate FAISS index of human_escalation trigger embeddings
escalation_dim = human_escalation_matrix.shape[1]
escalation_index = faiss.IndexFlatIP(escalation_dim)
escalation_index.add(human_escalation_matrix)




def attachment_handler(message):
    """
    Inputs
    message: dict of message info (will be from Firestore)

    Returns
    attachment_type: either None, 'audio', or 'unsupported'
    user_content: None, or str of transcribed user_content
    """

    attachment = message.get('attachment', None)


    if not attachment:
        return None, message.get('content')

    if attachment["type"] != "audio":
        return 'unsupported', None

    # Now type must be audio
    url = attachment["url"]
    try:
        http_response= requests.get(url, headers={"D360-API-KEY": dialog360_key}, timeout=20)
        http_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Unable to download audio at {url}: {e}")
        raise

    
    # Create temporary audio file
    audio_file = BytesIO(http_response.content)
    audio_file.name = "audio.ogg"

    # Get audio transcription
    transcript = openai_client.audio.transcriptions.create(
        model=transcription_model,
        file=audio_file,
        include=["logprobs"]
    )

    avg_token_confidence = sum(np.exp(lp.logprob) for lp in transcript.logprobs) / len(transcript.logprobs)
    if avg_token_confidence < p_audio:
        user_content = None
    else:
        user_content = transcript.text
    return 'audio', user_content


def clean_message(message):
    # Lowercase + trim
    cleaned = message.lower().strip()
    # Remove punctuation (everything except letters/numbers/whitespace)
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    # Remove the word 'olivia' wherever it appears
    cleaned = re.sub(r"\bolivia\b", "", cleaned)
    # Collapse multiple spaces into one + trim
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_msg_greeting(cleaned_message):
    for lang in supported_languages:
        if cleaned_message in greeting_triggers.get(lang, []):
            # We have detected a greeting
            return True, lang

    return False, None

def is_msg_gratitude(cleaned_message):
    for lang in supported_languages:
        if cleaned_message in gratitude_triggers.get(lang, []):
            # We have detected gratitude

            return True, lang

    return False, None

def detect_set_lang(user_content, db_lang, db_phone, doc_ref):
    """
    Determines or falls back to a user's language.

    Inputs:
        user_content (str or None): Text input from the user
        doc_ref: google.cloud.firestore_v1.document.DocumentReference used for writing lang to db


    Returns:
        lang (str): Language code (e.g., 'en', 'es', etc.)

    Also updates the lang in Firestore if a different, supported language is detected.
    """

    # Case 1: No message content — fallback only
    if not user_content:
        if db_lang:
            return db_lang
        else:
            prefix = db_phone[:2] if db_phone else ""
            lang = phone_to_language.get(prefix, "en")
            doc_ref.update({"meta.lang": lang})
            return lang

    # Case 2: Attempt language detection
    try:
        lang_detection = translate_client.detect_language(
            parent="projects/notifications-service-82362/locations/global",
            content=user_content,
            mime_type="text/plain"
        )

        best_match = lang_detection.languages[0]
        detected_lang = best_match.language_code
        is_confident = best_match.confidence >= p_langdetect

        if is_confident and detected_lang in supported_languages:
            if detected_lang != db_lang:
                doc_ref.update({"meta.lang": detected_lang})
            return detected_lang

        else:
            # Fallback path
            if db_lang:
                return db_lang
            else:
                prefix = db_phone[:2] if db_phone else ""
                lang = phone_to_language.get(prefix, "en")
                doc_ref.update({"meta.lang": lang})
                return lang

    # API fails: Fallback
    except Exception as e:
        logging.warning(f"Language detection failed: {e}")
        if db_lang:
            return db_lang
        else:
            prefix = db_phone[:2] if db_phone else ""
            lang = phone_to_language.get(prefix, "en")
            doc_ref.update({"meta.lang": lang})
            return lang
        




def moderate_message(user_content):
    """
    Inputs
    user_content (str): The message to be checked for moderation

    Returns
    list[str] or None: List of flagged moderation categories if flagged, else None
    """

    moderation_response = openai_client.moderations.create(
        model=moderation_model,
        input=user_content
    ).results[0]

    if not moderation_response.flagged:
        return None
    
    flagged_categories = [cat for cat, flagged in moderation_response.categories.model_dump().items() if flagged]
    logging.info(f"Message: {user_content}, flagged by moderation system. Flagged categories: {flagged_categories}")
    
    return flagged_categories





def generate_embedding(text):
    """
    Inputs
    text: string for embedding

    Returns
    embedding: a 1xn numpy array of text embedded using our chosen OpenAI embedding model
    """
    try:
        vector = openai_client.embeddings.create(input=text, model=embedding_model).data[0].embedding
        return np.array([vector], dtype=np.float32)
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        raise



def knn_search(embedding, index, k, p_cosine_min):
    # Potentially explore different search modules
    """
    Inputs
    embedding: numpy array of embedded text
    index: FAISS index
    p_cosine_min: minimum threshold for 'near' neighbours
    K: number of nearest neighbours to find

    Returns
    indices: list of indices of near enough neighbours (in descending order by cosine similarity)

    This function uses inner product to compute cosine similarity, so relies on vectors being normalised.
    """

    cosines, indices = index.search(embedding, k)

    return [idx for idx, sim in zip(indices[0], cosines[0]) if sim >= p_cosine_min]





def rewrite_query_update_summary(user_content, thread, summary):
    """
    Inputs
    user_query (str)
    thread (array): array of dicts (could be empty array)
    summary (str): string generated conversation summary

    Returns
    Also updates Firestore with new summary
    """

    if not summary:
        summary = ""

    # At regular intervals rewrite summary from last 10
    if len(thread) % restart_summary_every == 0:
        base = thread[-10:]
    else:
        base = thread[-3:]

    convo_plus = base + [{"role": "user", "content": user_content}]
    
    num_turns = len(convo_plus)
    convo_flattened = "\n".join([f"{m['role']}: {m['content']}" for m in convo_plus])

    final_summary_prompt = summary_rewrite_template.format(
        summary=summary,
        num_turns=str(num_turns),
        flattened_message_block=convo_flattened
    )

    # Final updated summary
    updated_summary = openai_client.chat.completions.create(
        model=summary_query_model,
        messages= [
            {
                "role": "user",
                "content": final_summary_prompt
            }
        ],
        temperature=0,
        top_p=1
    ).choices[0].message.content.strip()

    if updated_summary.startswith("TOPIC_SHIFT:"):
        logging.info("Summariser GPT detected topic shift")
        topic_shift = True
        updated_summary = updated_summary.replace("TOPIC_SHIFT:", "", 1).strip()
    else:
        topic_shift = False

    assistant_msg = next((m["content"] for m in reversed(base) if m["role"] == "assistant"), "")

    final_query_prompt = query_rewrite_user_template.format(
        summary=updated_summary,
        assistant_msg=assistant_msg,
        user_msg=user_content
    )


    optimised_query = openai_client.chat.completions.create(
        model=summary_query_model,
        messages= [
            {
                "role": "system",
                "content": query_rewrite_system_prompt
            },
            {
                "role": "user",
                "content": final_query_prompt
            }
        ],
        temperature=0,
        top_p=1
    ).choices[0].message.content

    logging.info(f"User query optimised for retrieval: {optimised_query}")

    return updated_summary, optimised_query, topic_shift




def gpt_responder(information_array, summary, assistant_msg, user_msg, lang):
    """
    Inputs
    information_array (array): containing strings of retreived information
    summary (str): GPT generated summary

    Returns
    (str) generated GPT response
    """

    context = '\n\n'.join(information_array) or "No supporting information was found."

    permitted_topics_str = ", ".join(s.lower() for s in permitted_topics)
    banned_topics_str = ", ".join(s.lower() for s in banned_topics)

    response = openai_client.chat.completions.create(
        model=assistant_model,
        messages = [
                {"role": "system", "content": generate_response_prompt.format(lang_code=lang, summary=summary, information=context, permitted_topics=permitted_topics_str, banned_topics=banned_topics_str)},
                # Previous messages, in correct order
                {"role": "assistant", "content": assistant_msg},
                {"role": "user", "content": user_msg}
        ],
        temperature=0,
        top_p=1
    ).choices[0].message.content

    logging.info("GPT response generated")
    return response



def create_assistant_entry(response):
    return {
        "ID": str(uuid.uuid4()).upper(),
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
    }



@functions_framework.http
def receive_and_send(request):
    # Handle CORS preflight
    if request.method == "OPTIONS":
        return ("", 204, CORS_HEADERS)

    try:
        data = request.get_json()
        doc_id = data["ID"]
    except (TypeError, KeyError):
        return (
            json.dumps({"error": "'No document ID passed"}),
            400,
            {**CORS_HEADERS, "Content-Type": "application/json"},
        )


    
    # Get all relevant info from database
    doc_ref = db_olivia.collection("conversations").document(doc_id)
    snapshot = doc_ref.get()

    if snapshot.exists:
        doc_dict = snapshot.to_dict()
    else:
        logging.error(f"Unable to find conversation document, ID: {doc_id}")
        return json.dumps({"error": f"Conversation document not found: {doc_id}"}), 404


    meta = doc_dict["meta"]

    db_lang = meta.get("lang", None)
    db_phone = meta.get("phone", None)

    context = doc_dict["context"]

    # May be null
    summary = context["summary"]
    # May be empty array
    thread = context["thread"]

    dialog = doc_dict["dialog"]
    if dialog is None:
        logging.error(f"No dialog found for user: {db_phone}")
        return json.dumps({"error": f"No dialog found for user: {db_phone}"}), 400


    # Get the latest user message
    user_msg = next(d for d in reversed(dialog) if d.get('role') == 'user')
    user_msg_ID = user_msg["ID"]
    user_msg_content = user_msg["content"]

    # And assistant message
    assistant_msg = next((d for d in reversed(dialog) if d.get('role') == 'assistant'), None)
    assistant_msg_content = assistant_msg["content"] if assistant_msg else ""


    user_msg_attachment, user_msg_content = attachment_handler(user_msg)


    if user_msg_attachment == "unsupported":
        lang = detect_set_lang(user_msg_content, db_lang, db_phone, doc_ref)
        response = unsupported_media_responses[lang]
        assistant_entry = create_assistant_entry(response)

        dialog.append(assistant_entry)
        doc_ref.update({"dialog": dialog})
        return "", 200


    if not user_msg_content:
        lang = detect_set_lang(user_msg_content, db_lang, db_phone, doc_ref)
        response = not_understood_responses[lang]
        assistant_entry = create_assistant_entry(response)

        dialog.append(assistant_entry)
        doc_ref.update({"dialog": dialog})
        return "", 200

    cleaned_message = clean_message(user_msg_content)

    greeting, lang = is_msg_greeting(cleaned_message)
    if greeting:
        response = greeting_responses[lang]
        assistant_entry = create_assistant_entry(response)

        dialog.append(assistant_entry)
        doc_ref.update({"dialog": dialog})
        return "", 200
    
    gratitude, lang = is_msg_gratitude(cleaned_message)
    if gratitude:
        response = gratitude_responses[lang]
        assistant_entry = create_assistant_entry(response)

        dialog.append(assistant_entry)
        doc_ref.update(
            {
                "dialog": dialog,
                "context.thread": [],
                "context.summary": None
            }
        )
        return "", 200
    
    # FINAL CHECK FOR IF INPUT IS JUST OLIVIA, IF SO CONTINUE TO DETECT LANGUAGE AND GIVE A GREETING RESPONSE

    lang = detect_set_lang(user_msg_content, db_lang, db_phone, doc_ref)

    moderation_flags = moderate_message(user_msg_content)


    if moderation_flags:
        if set(moderation_flags) & set(moderation_self_harm_categories):
            response = self_harm_responses[lang]
            assistant_entry = create_assistant_entry(response)

            dialog.append(assistant_entry)
            doc_ref.update({"dialog": dialog})
            return "", 200
        
        else: 
            response = flagged_content_responses[lang]
            assistant_entry = create_assistant_entry(response)

            dialog.append(assistant_entry)
            doc_ref.update({"dialog": dialog})
            return "", 200



    user_entry = {
        "ID": user_msg_ID,
        "role": "user",
        "content": user_msg_content,
        "timestamp": user_msg["timestamp"]
    }

    content_embedding = generate_embedding(user_msg_content)

    if knn_search(content_embedding, escalation_index, 1, p_human_escalation):
        # We are certain they want to speak to human
        response = hard_escalate_responses[lang]
        assistant_entry = create_assistant_entry(response)
        
        dialog.append(assistant_entry)

        doc_ref.update(
            {
            "dialog": dialog,
            "escalate": True
            }
        )
        return "", 200

    
    # USED TO PASS DIALOG INSTEAD OF THREAD, CHECK BEFORE PUSHING
    updated_summary, optimised_query, topic_shift = rewrite_query_update_summary(user_msg_content, thread, summary)


    optimised_query_embedding = generate_embedding(optimised_query)
    search_indices = knn_search(optimised_query_embedding, info_index, k, p_cosine_min)
    search_results = info_df.loc[search_indices, f'information_{lang}'].tolist()


    response = gpt_responder(search_results, updated_summary, assistant_msg_content, user_msg_content, lang)

    # Did the GPT deem it a banned topic?
    if response == "BANNED_94736":
        logging.info("GPT identified banned content")
        response = flagged_content_responses[lang]
        assistant_entry = create_assistant_entry(response)
        dialog.append(assistant_entry)
        # Add to dialog array but not thread. Don't update summary.
        doc_ref.update({"dialog": dialog})
        return "", 200
 
    elif response == "UNKNOWN_45783":
        logging.info("GPT identified irrelevant, or hard-to-understand content")
        response = not_understood_responses[lang]
        assistant_entry = create_assistant_entry(response)
        dialog.append(assistant_entry)
        # Add to dialog array but not thread. Don't update summary.
        doc_ref.update({"dialog": dialog})
        return "", 200
    
    elif response == "RESOLVED_72859":
        logging.info("GPT determined the issue resolved")
        response = resolved_responses[lang]
        assistant_entry = create_assistant_entry(response)
        dialog.append(assistant_entry)
        # Add to dialog array, clear thread and summary.
        doc_ref.update(
            {
                "dialog": dialog,
                "context.thread": [],
                "context.summary": None
            }
        )
        return "", 200


    # Clear the thread if topic has changed
    if topic_shift:
        thread = []

    # Now message has passed all checks and response has been generated
    assistant_entry = create_assistant_entry(response)

    thread.extend([user_entry, assistant_entry])
    dialog.append(assistant_entry)

    monitor_thread = False
    if len(thread) == 8:
        monitor_thread = True
    
    doc_ref.update(
        {
            "dialog": dialog,
            "context.thread": thread,
            "context.summary": updated_summary,
            "monitor_thread": monitor_thread
        }
    )
    return "", 200