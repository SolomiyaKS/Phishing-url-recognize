import logging
import joblib
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters


model = joblib.load("models/phishing_model2.pkl")
vectorizer = joblib.load("models/vectorizer2.pkl")
scaler = joblib.load("models/caler2.pkl")


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привіт! Я бот для перевірки URL на фішинг. Оберіть опцію:\n"
        "/instruction - Як користуватися ботом\n"
        "/check - Надішліть URL для перевірки"
    )


async def check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Надішліть URL для перевірки.")


async def instruction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Щоб перевірити URL:\n"
        "1. Оберіть команду /check.\n"
        "2. Відправте URL, який потрібно перевірити.\n"
        "Я поверну результат: фішинговий ⚠️ чи легітимний ✅, а також покажу впевненість у прогнозі."
    )


async def classify_url(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_input = update.message.text

    if not user_input:
        await update.message.reply_text("Будь ласка, надайте URL для перевірки.")
        return


    input_vec = vectorizer.transform([user_input])

    input_scaled = scaler.transform(input_vec)


    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0] 

    if prediction == 'legitimate':
        class_index = 0 
    else:
        class_index = 1 

    confidence = probabilities[class_index] * 100 

    if prediction == 'legitimate':
        result = "Legitimate ✅"
    else:
        result = "Phishing ⚠️"
    
    response = (
        f"Result: {result}\n"
        f"Confidence: {confidence:.2f}%"
    )

    await update.message.reply_text(response)

def main():
    telegram_token = "YOUR_TOKEN"

    application = Application.builder().token(telegram_token).build()

    # Додавання команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("instruction", instruction))
    application.add_handler(CommandHandler("check", check))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, classify_url))

    application.run_polling()

if __name__ == "__main__":
    main()
