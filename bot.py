import atexit
import os
import pickle
from datetime import datetime, timedelta
from io import BytesIO

import requests
import schedule
import telebot
from PIL import Image

from config import BOT_TOKEN
from network import ClipNetwork

bot = telebot.TeleBot(BOT_TOKEN)
network = ClipNetwork()
save_dir = 'saves'
os.makedirs(save_dir, exist_ok=True)

@bot.message_handler(content_types=['photo'])
def handle_photos(message):
    photos = message.photo

    photo = photos[-1]
    file_id = photo.file_id

    file_info = bot.get_file(file_id)
    file_url = f'https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}'

    response = requests.get(file_url)

    image = None
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
    else:
        print("Failed to download image.")

    image_embeddings = network.transform_images([image])
    chat_id = message.chat.id
    if chat_id not in embeddings:
        embeddings[chat_id] = {
            'embeddings': [],
            'message_ids': []
        }

    embeddings[chat_id]['embeddings'].extend(image_embeddings)
    embeddings[chat_id]['message_ids'].extend([message.message_id] * len(image_embeddings))


@bot.message_handler(commands=['find_image'])
def handle_find_image_command(message):
    chat_id = message.chat.id
    if chat_id not in embeddings:
        bot.send_message(chat_id, text='Have no photo from this chat')
        return

    command_text = message.text.split(" ", 1)
    command_text = command_text[1].split(" ")

    top_k = command_text[-1]

    if top_k.isdigit():
        query = " ".join(command_text[:-1])
        top_k = int(top_k)
    else:
        query = " ".join(command_text)
        top_k = 1

    if len(embeddings[chat_id]['message_ids']) < top_k:
        bot.send_message(chat_id, text='The number is more than images length!')
        return

    ids = network.find_top_k(query, embeddings[chat_id]['embeddings'], top_k)
    for id in ids:
        bot.forward_message(chat_id, chat_id, embeddings[chat_id]['message_ids'][id])


def save_embeddings():
    today = datetime.now().strftime("%d.%m.%Y")
    filename = os.path.join(save_dir, f"{today}-embeddings.pkl")

    with open(filename, 'wb') as file:
        pickle.dump(embeddings, file)
    print(f"Embeddings saved: {filename}")

    for file in os.listdir(save_dir):
        if file.endswith('-embeddings.pkl') and file != f"{today}-embeddings.pkl":
            os.remove(os.path.join(save_dir, file))
            print(f"Previous day embeddings deleted: {file}")


def load_embeddings():
    embedding_files = [f for f in os.listdir(save_dir) if f.endswith('-embeddings.pkl')]

    if not embedding_files:
        print("No files for loading (it must ends with -embeddings.pkl).")
        return {}

    first_file = embedding_files[0]
    filename = os.path.join(save_dir, first_file)

    with open(filename, 'rb') as file:
        embeddings = pickle.load(file)
    print(f"File was loaded: {filename}")
    return embeddings


embeddings = load_embeddings()  # {chat_id : {'embeddings' : [], 'message_ids' : []}}
atexit.register(save_embeddings)
schedule.every().day.at("00:00").do(save_embeddings)
bot.infinity_polling()
