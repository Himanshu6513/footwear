import json, openai

def create_product_embeddings(input_path, output_path, api_key, model="text-embedding-3-small"):
    openai.api_key = api_key
    data = json.load(open(input_path))
    for item in data:
        try:
            item['embedding'] = openai.embeddings.create(input=item['description'], model=model).data[0].embedding
        except Exception:
            item['embedding'] = None
    json.dump(data, open(output_path, 'w'), indent=2)


api_key = "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"
create_product_embeddings(
    input_path='/content/footwear_metadata.json',
    output_path='/content/footwear_with_embeddings.json',
    api_key=api_key
)