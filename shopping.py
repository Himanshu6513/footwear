from fastapi import FastAPI, Request
from fastapi.responses import Response
import requests, json, os, time, asyncio
from openai import OpenAI
from google.cloud import storage
from datetime import timedelta
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import numpy as np
import json
import time
app = FastAPI()

# === Configuration ===
GCS_BUCKET_NAME = "clothes_shop"
GCS_CREDENTIALS_FILE = "twilio-440407-9049c36e31b5.json"
PRODUCT_JSON_PATH = "footwear_metadata.json"
client = OpenAI(api_key="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")
TWILIO_SID = "ACc3b466139e779e862c4f545bd6e19d94"
TWILIO_AUTH = "0587b58274800f397550c85b621ab921"
TWILIO_NUMBER = "whatsapp:+919319837618"
embedded_json_path="footwear_with_embeddings.json"
# === Google Cloud Storage Setup ===
storage_client = storage.Client.from_service_account_json(GCS_CREDENTIALS_FILE)
bucket = storage_client.bucket(GCS_BUCKET_NAME)

# === In-memory sessions ===
sessions = {}  # phone -> { 'history': [...], 'last_active': timestamp }

# === Utilities ===
def generate_detailed_product_description(client, combined_text: str) -> str:
    system_prompt = (
        "You are a product assistant. The user input may describe an image, caption, or sentence referring to an item.\n\n"
        "If the input is related to footwear, identify the specific type such as 'sports shoes', 'crocs', 'formal leather shoes', etc. "
        "If not footwear, determine the most relevant product category (like 'hat', 'backpack', 'watch', 'dress').\n\n"
        "Then, based on the input, describe the product in **detail**:\n"
        "- Mention **specific characteristics** such as exact color tone, material type, sole design, pattern, straps, usage context (e.g., office, sports, casual wear), etc.\n"
        "- Avoid general terms like 'color', 'design', or 'style' alone—describe them explicitly (e.g., 'light grey mesh upper with neon green accents').\n"
        "- Do NOT use vague phrases like 'something similar' or 'like this'. Be **clear and descriptive**.\n\n"
        "Always return the response in this format:\n"
        "I am looking for [product name] that matches **at least one** of the following aspects: [detailed aspect 1], [detailed aspect 2], [detailed aspect 3], etc.\n\n"
        "Never include explanations, just the final descriptive sentence."
    )

    user_prompt = f"Here is the combined input: \"{combined_text.strip()}\""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"OpenAI API error: {e}"

def call_openai_chat(phone_number):
    chat_history=load_session(phone_number)
    language = detect_latest_language(chat_history)
    user_intention=detect_latest_intention(chat_history)
    print(user_intention)
    system_instruction = f"""
You are an AI assistant for XYZ Footwear Store. Your role is to help users explore and select the right type of footwear based on their needs and preferences.

If the conversation has just started (e.g., user said "Hi", "Hello", etc.), then just introduce yourself as the assistant for XYZ Footwear Store and briefly explain how you can help.

Key behavior guidelines:
- If the user talks about any unrelated topic other than footwear, gently steer the conversation back to selecting footwear.
- If the user asks about the product details or products shown earlier, help them continue naturally or provide relevant information.
- If the user has not accurately defined the kind of product they have been looking for, politely ask them to clarify their needs so you can assist them better.

The conversation is in: **{language}**

Recent user intention:
"{user_intention}"
"""

    decision_prompt = f"""
{system_instruction}

Based on the recent user intention, return a single character indicating the assistant's next step:

- f → if the user says anything that implies interest in any kind of footwear — including mentioning specific categories such as shoes, sandals, slippers, formal shoes, kids shoes, men's or ladies footwear — even if vaguely or indirectly.
- p → if the user clearly wants to see more items similar to what has already been shown or discussed.
- n → if the user's intention is unclear or unrelated to footwear, or cannot be confidently classified into 'f' or 'p'.

Return only one character: f, p, or n.
"""

    action_token = None
    for _ in range(5):
        res = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a decision-making assistant."},
                {"role": "user", "content": decision_prompt}
            ]
        )
        result = res.choices[0].message.content.strip().lower()
        print(result)
        if result in ["f", "p", "n"]:
            action_token = result
            break
    
    if action_token is None:
        action_token = "n"  # Default to 'n' after 5 unsuccessful attempts

    # --- Action: Find and pitch matching footwear ---
    if action_token == "f":
        relevant_product = find_relevant_from_embedded_json(embedded_json_path=embedded_json_path,client=client,user_intent=user_intention)
        products = json.loads(relevant_product)
        if not products:
            text_to_send=translate_to_language(text="Sorry, could not find a relevant product as per your requirement",target_lang=language,client=client)
            reply = {"role": "assistant", "content": text_to_send}
            update_session(phone_number,reply)
            send_twilio_message(to=phone_number, text=text_to_send)
            return
        else:
            relevant_product,leftover_product=filter_and_generate_pitches(products=products,user_requirement=user_intention,client=client,language=language)
            if not  relevant_product:
                text_to_send=translate_to_language(text="Sorry, could not find a relevant product as per your requirement",target_lang=language,client=client)
                reply = {"role": "assistant", "content": text_to_send}
                update_session(phone_number,reply)
                send_twilio_message(to=phone_number, text=text_to_send)
                return
            else:
                user_intent_blob_path = f"user_intent/{phone_number}.txt"
                user_intent_blob = bucket.blob(user_intent_blob_path)
                user_intent_blob.upload_from_string(user_intention, content_type='text/plain')
                summary_text = "Here are the footwear along with their details which are currently under consideration:\n"
                text_to_send=translate_to_language(text="Here are some of the product as per your requirment.Do let us know if you are looking anything more of this kind or anything else:-",target_lang=language,client=client)
                reply = {"role": "assistant", "content": text_to_send}
                update_session(phone_number,reply)
                send_twilio_message(to=phone_number, text=text_to_send)
                time.sleep(2)
                for product in relevant_product:
                    caption = product.get("caption_new", "")
                    persuasive = product.get("persuasive", "")
                    image_url = product.get("image_url", None)
                    caption_original = product.get("caption", "")
                    description = product.get("description", "")
                    summary_text += f"\n• {caption_original}\n{description}\n"
                    combined_text = f"*{caption}*\n\n{persuasive}" if caption or persuasive else None
                    send_twilio_message(to=phone_number,text=combined_text,image_url=image_url)
                    time.sleep(2)
                    reply = {"role": "assistant","content": combined_text,"image_url": image_url}
                    update_session(phone_number, reply)
                blob_path = f"product/{phone_number}.txt"
                blob = bucket.blob(blob_path)
                blob.upload_from_string(summary_text, content_type='text/plain')
                if leftover_product:
                    leftover_blob_path = f"remaining/{phone_number}.json"
                    leftover_blob = bucket.blob(leftover_blob_path)
                    leftover_json = json.dumps(leftover_product, ensure_ascii=False, indent=2)
                    leftover_blob.upload_from_string(leftover_json, content_type='application/json')
                return

    # --- Action: Show more similar products ---
    elif action_token == "p":
        leftover_blob_path = f"remaining/{phone_number}.json"
        leftover_blob = bucket.blob(leftover_blob_path)
        if leftover_blob.exists():
            leftover_json_str = leftover_blob.download_as_text()
            leftover_data = json.loads(leftover_json_str)
            leftover_blob.delete()
            user_intent_blob_path = f"user_intent/{phone_number}.txt"
            user_intent_blob = bucket.blob(user_intent_blob_path)
            if user_intent_blob.exists():
                user_intention = user_intent_blob.download_as_text()
            else:
                user_intention = ""
            relevant_product,leftover_product=filter_and_generate_pitches(products=leftover_data,user_requirement=user_intention,client=client,language=language)
            if not  relevant_product:
                text_to_send=translate_to_language(text="Sorry, could not find more  product as per your requirement",target_lang=language,client=client)
                reply = {"role": "assistant", "content": text_to_send}
                update_session(phone_number,reply)
                send_twilio_message(to=phone_number, text=text_to_send)
                return
            else:
                product_blob_path = f"product/{phone_number}.txt"
                product_blob = bucket.blob(product_blob_path)
                if product_blob.exists():
                    summary_text = product_blob.download_as_text()
                else:
                    summary_text = ""
                text_to_send=translate_to_language(text="Here are some more products as per your requirment.Do let us know if you are looking anything more of this kind or anything else:-",target_lang=language,client=client)
                reply = {"role": "assistant", "content": text_to_send}
                update_session(phone_number,reply)
                send_twilio_message(to=phone_number, text=text_to_send)
                time.sleep(2)
                for product in relevant_product:
                    caption = product.get("caption_new", "")
                    persuasive = product.get("persuasive", "")
                    image_url = product.get("image_url", None)
                    caption_original = product.get("caption", "")
                    description = product.get("description", "")
                    summary_text += f"\n• {caption_original}\n{description}\n"
                    combined_text = f"*{caption}*\n\n{persuasive}" if caption or persuasive else None
                    send_twilio_message(to=phone_number,text=combined_text,image_url=image_url)
                    time.sleep(2)
                    reply = {"role": "assistant","content": combined_text,"image_url": image_url}
                    update_session(phone_number, reply)
                blob_path = f"product/{phone_number}.txt"
                blob = bucket.blob(blob_path)
                blob.upload_from_string(summary_text, content_type='text/plain')
                if leftover_product:
                    leftover_blob_path = f"remaining/{phone_number}.json"
                    leftover_blob = bucket.blob(leftover_blob_path)
                    leftover_json = json.dumps(leftover_product, ensure_ascii=False, indent=2)
                    leftover_blob.upload_from_string(leftover_json, content_type='application/json')
                return
        else:
            text_to_send=translate_to_language(text="Sorry,we currently dont have more product as per your requirment ",target_lang=language,client=client)
            reply = {"role": "assistant", "content": text_to_send}
            update_session(phone_number,reply)
            send_twilio_message(to=phone_number, text=text_to_send)
            return

    # --- Action: Clarify or redirect ---
    else:
        product_blob_path = f"product/{phone_number}.txt"
        product_blob = bucket.blob(product_blob_path)
        if product_blob.exists():
            summary_text = product_blob.download_as_text().strip()
        else:
            summary_text = ""
        summary_section = f"\n{summary_text}\n" if summary_text else ""
        natural_prompt = f"""
                          {system_instruction}
                          {summary_section}
                        Now generate the next assistant message in response to the user. Language: {language}.
                        Conversation so far:
                         {chat_history}
                          """
        res2 = client.chat.completions.create(model="gpt-4o",temperature=0.7,messages=[{"role": "system", "content": "You are a helpful and polite assistant."},{"role": "user", "content": natural_prompt}])
        text_to_send = res2.choices[0].message.content.strip()
        reply = {"role": "assistant", "content": text_to_send}
        update_session(phone_number,reply)
        send_twilio_message(to=phone_number, text=text_to_send)
        return

def detect_latest_language(chat_history):
    prompt = f"""You will be given a conversation between a user and an assistant.

Identify the language used in the latest message from the user.

Respond with just the name of the language as one word.
Example response: French

Here is the conversation:
{extract_chat_text(chat_history)}
"""

    res = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    lang = res.choices[0].message.content.strip().capitalize()
    allowed = ["Hindi", "English", "Tamil"]
    return lang if lang in allowed else "English"
def extract_chat_text(chat_history):
    return "\n".join(
        f"{e['role'].capitalize()}: {' '.join([c['text'] if c['type'] == 'text' else '' for c in e['content']]) if isinstance(e['content'], list) else e['content']}"
        for e in chat_history
    )

def detect_latest_intention(chat_history):
    prompt = f"""
You are analyzing a conversation between a user and an assistant.

Your job is to detect the user's **current intent**, especially if they are looking for any kind of **footwear** — such as shoes, sandals, slippers, crocs, sneakers, boots, heels, etc.

🗣️ The user may speak in **any language**, including Hindi, English, Hinglish, French, Spanish, or a mix. You must detect intent regardless of language.

Internally, **translate or interpret the user's message to understand it**, but always reply using one of the response formats below in **English only**.

Respond with one of the following:

1. ✅ "User is looking for: [type of footwear] that is [color/style/material], in size [size], intended for [purpose]"  
→ Footwear type is **mandatory**. Add other attributes only if clearly mentioned.

2. 🔁 "User is looking for more of the kind of product shown earlier, likely similar in type, color, and style."  
→ If the user refers to previous examples and asks for similar ones.

3. ❌ "User is currently talking about something unrelated to footwear."  
→ If the user is off-topic.

4. ➡️ "User has not clearly defined the product they are looking for."  
→ If no footwear type is mentioned.

5.🔍 "User is looking for [type of footwear] that matches at least one of the following aspects: style, color scheme, type, material, brand, design features, or other visual characteristics."
→  Use if the footwear type is clearly mentioned, and the user is seeking similarity in at least one specific aspect (e.g., style, color, or material, etc).

Rules:
- Use **contextual understanding and internal translation** — e.g., recognize "jute" as shoes (Hindi), "zapatos" (Spanish), "chaussures" (French), etc.
- Do **not assume** any missing info. Only mention attributes explicitly stated.
- Even if the user switches topics, detect the **latest intent** expressed in the conversation.

Conversation:
{extract_chat_text(chat_history)}
"""

    res = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content.strip()

def filter_and_generate_pitches(products, user_requirement, client, language="English", model="gpt-4o", max_matches=5):
    matched_products = []

    def get_match_response(prompt, retries=5):
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful product relevance assistant. "
                                "Return '1' if the product is reasonably relevant to the user's requirement, "
                                "even if not a perfect match. Return '0' if clearly unrelated. No explanation."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                result = response.choices[0].message.content.strip()
                if result in ("1", "0"):
                    return result
            except Exception as e:
                print(f"[Match Retry {attempt + 1}] Error: {e}")
                time.sleep(2)
        return "0"

    def get_persuasive_text(product, retries=5):
        prompt = (
            f"User need: {user_requirement}\n\n"
            f"Product:\n"
            f"Caption: {product['caption']}\n"
            f"Description: {product['description']}\n\n"
            f"Write a short and simple message (under 150 characters) telling a regular shopper why this product is a good choice. "
            f"Use easy language anyone can understand. Be friendly, clear, and direct. "
            f"Focus on comfort, material, color, price, or anything that fits the user’s need. "
            f"Avoid technical or marketing terms. "
            f"Write the message in {language}."
        )
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant writing short product pitches."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.6,
                    max_tokens=200
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[Pitch Retry {attempt + 1}] Error: {e}")
                time.sleep(2)
        return "Pitch unavailable."

    def get_caption_translation(caption, language, retries=5):
        prompt = (
            f"Translate or rewrite this product caption into {language}. "
            f"Keep it short and natural: '{caption}'"
        )
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that translates product captions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=100
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[Caption Retry {attempt + 1}] Error: {e}")
                time.sleep(2)
        return caption  # fallback to original

    i = 0
    while i < len(products) and len(matched_products) < max_matches:
        product = products[i]
        prompt = (
            f"User requirement:\n{user_requirement}\n\n"
            f"Product Caption: {product.get('caption', '')}\n"
            f"Product Image URL: {product.get('image_url', '')}\n"
            f"Product Description:\n{product.get('description', '')}\n\n"
            f"Does this product match the user's need? Respond only with '1' or '0'."
        )
        match = get_match_response(prompt)
        if match == "1":
            product["match"] = "1"
            product["persuasive"] = get_persuasive_text(product)
            product["caption_new"] = get_caption_translation(product.get("caption", ""), language)
            matched_products.append(product)
            products.pop(i)
        else:
            products.pop(i)

    return matched_products, products
# ---- Cosine similarity for ranking ----
def cosine_similarity(vec1, vec2):
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---- STEP 1: Get Top-K relevant products based on embeddings ----
def find_relevant_from_embedded_json(embedded_json_path, client, user_intent, top_k=20, min_similarity=0.10):
    with open(embedded_json_path, 'r') as f:
        products = json.load(f)

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=user_intent
    )
    user_embedding = response.data[0].embedding

    scored_products = []
    for product in products:
        embedding = product.get("embedding")
        if not embedding:
            continue
        similarity = cosine_similarity(user_embedding, embedding)
        if similarity > min_similarity:
            scored_products.append((similarity, product))

    scored_products.sort(key=lambda x: x[0], reverse=True)

    top_products = []
    for score, product in scored_products[:top_k]:
        clean_product = {k: v for k, v in product.items() if k != "embedding"}
        clean_product["similarity"] = round(score, 3)
        top_products.append(clean_product)

    return json.dumps(top_products, indent=2)  #  returns string!


def translate_to_language(text, target_lang, client):
    prompt = f"""
Translate the following message to {target_lang}.
The input text can be in any language.
Your response should only include the translated version, without any explanation.

Text:
{text}
""".strip()

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Translation error: {str(e)})"

def upload_image_to_gcs(image_url: str, filename_base: str) -> str:
    response = requests.get(image_url, auth=(TWILIO_SID, TWILIO_AUTH))
    if response.status_code != 200:
        raise Exception(f"Failed to fetch image from Twilio. Status code: {response.status_code}")

    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        raise ValueError(f"Unsupported media type: {content_type}")

    try:
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"Image could not be identified or converted: {e}")

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    blob = bucket.blob(f"user_image/{filename_base}.jpeg")
    blob.upload_from_file(buffer, content_type="image/jpeg")
    return f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/user_image/{filename_base}.jpeg"

def load_session(phone: str):
    now = time.time()
    if phone in sessions:
        sessions[phone]['last_active'] = now
        return sessions[phone]['history']

    blob = bucket.blob(f"chat/{phone}.json")
    if blob.exists():
        blob.reload()
        if blob.updated and (time.time() - blob.updated.timestamp() < 7200):
            history = json.loads(blob.download_as_text())
            sessions[phone] = { 'history': history, 'last_active': now }
            return history
        else:
            blob.delete()

    sessions[phone] = { 'history': [], 'last_active': now }
    return sessions[phone]['history']

def save_session_to_gcs(phone: str):
    if phone in sessions:
        history = sessions[phone]['history']
        blob = bucket.blob(f"chat/{phone}.json")
        blob.upload_from_string(json.dumps(history), content_type="application/json")
        del sessions[phone]

def update_session(phone: str, message: dict):
    history = load_session(phone)
    history.append(message)
    sessions[phone]['last_active'] = time.time()

def extract_text_and_images(reply_msg):
    if isinstance(reply_msg.content, str):
        return reply_msg.content, None
    elif isinstance(reply_msg.content, list):
        text = ""
        image_url = None
        for part in reply_msg.content:
            if part.get("type") == "text":
                text += part.get("text", "") + "\n"
            elif part.get("type") == "image_url":
                image_url = part.get("image_url", {}).get("url")
        return text.strip(), image_url
    return "", None
def call_openai_image_description(image_url: str):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert visual assistant. Your task is to analyze the provided image.\n\n"
                "If the image contains footwear, describe it using the following attributes:\n"
                "- Style (e.g., casual, sporty, formal)\n"
                "- Color(s)\n"
                "- Type (e.g., sneakers, sandals, boots)\n"
                "- Material (e.g., leather, canvas, mesh)\n\n"
                "If no footwear is found, provide a general image description instead.\n\n"
                "Always return the result in this exact format:\n\n"
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=500,
        temperature=0
    )

    return response.choices[0].message.content
def send_twilio_message(to, text=None, image_url=None):
    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json"
    data = {
        "From": TWILIO_NUMBER,
        "To": f"whatsapp:{to}"
    }
    if text:
        data["Body"] = text
    if image_url:
        data["MediaUrl"] = image_url
    requests.post(url, data=data, auth=(TWILIO_SID, TWILIO_AUTH))

@app.post("/whatsapp/")
async def whatsapp_webhook(request: Request):
    form = await request.form()
    from_number = form.get("From", "").split(":")[-1]
    body = form.get("Body", None)
    num_media = int(form.get("NumMedia", 0))

    image_url = None
    if num_media == 1:
        content_type = form.get("MediaContentType0")
        media_url = form.get("MediaUrl0")
        if content_type.startswith("image"):
            filename_base = f"{from_number}_{int(time.time())}"
            try:
                image_url = upload_image_to_gcs(media_url, filename_base)
            except Exception:
                return Response(status_code=204)
        else:
            return Response(status_code=204)

    if body and not image_url:
        update_session(from_number, {"role": "user", "content": body})
        call_openai_chat(from_number)
        return

    elif image_url and not body:
        description = call_openai_image_description(image_url).strip()
        print(description)
        user_text="If the description below refers to a footwear item, I’m looking for footwear  with features in the description below. If it doesn't, please summarize what the description is about:-"
        chat_history=load_session(from_number)
        language = detect_latest_language(chat_history)
        combined_message = f"{user_text}\n{description}"
        final_combined_text=generate_detailed_product_description(client=client, combined_text=combined_message)
        combined_message_translated=translate_to_language(text=final_combined_text,target_lang=language,client=client)
        message = {"role": "user","content": [{"type": "text", "text": combined_message_translated}]}
        update_session(from_number, message)
        call_openai_chat(from_number)
        return

    elif image_url and body:
        description = call_openai_image_description(image_url).strip()
        print(description)
        chat_history=load_session(from_number)
        language = detect_latest_language(chat_history)
        combined_message = f"{description}\n{body}"
        final_combined_text=generate_detailed_product_description(client=client, combined_text=combined_message)
        print(final_combined_text)
        combined_message_translated=translate_to_language(text=final_combined_text,target_lang=language,client=client)
        message = {"role": "user","content": [{"type": "text", "text": combined_message_translated}]}
        update_session(from_number, message)
        call_openai_chat(from_number)
        return
    else:
        return Response(status_code=204)

@app.on_event("startup")
async def cleanup_inactive_sessions():
    async def periodic_cleanup():
        while True:
            now = time.time()
            to_save = [phone for phone, data in sessions.items() if now - data['last_active'] > 900]
            for phone in to_save:
                save_session_to_gcs(phone)
            await asyncio.sleep(60)
    asyncio.create_task(periodic_cleanup())
