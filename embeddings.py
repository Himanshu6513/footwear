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


api_key = "#Your OpenAI API Key"
create_product_embeddings(
    input_path='/content/footwear_metadata.json',
    output_path='/content/footwear_with_embeddings.json',
    api_key=api_key

)
