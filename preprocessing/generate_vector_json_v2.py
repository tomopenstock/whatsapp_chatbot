import numpy as np
import openai
import pandas as pd
import json
from google.cloud import secretmanager
from google.cloud import storage
import uuid


secret_client = secretmanager.SecretManagerServiceClient()


# Build the resource name of the secret
secret_name = f"projects/1052202528756/secrets/OPENAI_API_KEY/versions/latest"

# Access the secret version
response = secret_client.access_secret_version(name=secret_name)
api_key = response.payload.data.decode("UTF-8")


# Initialise OpenAI API
client = openai.OpenAI(api_key = api_key)


# Create dataframe of information
with open("/Users/openstockltd/Documents/Chatbot/V2/preprocessing/olivia_information.json") as f:
    data = json.load(f)

    

embedding_model = data['models']['embedding_model']
information_df = pd.DataFrame(data["information"])
human_escalation_df = pd.DataFrame(data["human_escalation"])

metadata = {k: v for k, v in data.items() if k not in ["information", "human_escalation"]}

embedding_information = client.embeddings.create(input=information_df["information_en"].tolist(), model=embedding_model)
embedding_human_escalation = client.embeddings.create(input=human_escalation_df["text"].tolist(),model=embedding_model)
print("Text embeddings created.")


information_df["vector_embedding"] = [e.embedding for e in embedding_information.data]
information_df["id"] = [str(uuid.uuid4()) for _ in range(len(information_df))]

human_escalation_df["vector_embedding"] = [e.embedding for e in embedding_human_escalation.data]
human_escalation_df["id"] = [str(uuid.uuid4()) for _ in range(len(human_escalation_df))]


# Save JSON to files ready for use in production
output = metadata | {"human_escalation": human_escalation_df.to_dict(orient="records")} | {"information": information_df.to_dict(orient="records")}


with open("/Users/openstockltd/Documents/Chatbot/V2/preprocessing/olivia_information_vectors.json", "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print("olivia_information_vectors.json generated and saved.")