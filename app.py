import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from textblob import TextBlob
import matplotlib.pyplot as plt

@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis")
    emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
    emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
    emotion_pipeline = pipeline("text-classification", model=emotion_model, tokenizer=tokenizer, return_all_scores=False)
    return sentiment_model, emotion_pipeline

sentiment_model, emotion_pipeline = load_models()

st.title("📊 AI-Powered Social Media Sentiment & Engagement Assistant")
st.markdown("Analyze text sentiment, detect emotions, and chat with an empathetic AI assistant.")

mode = st.sidebar.radio("Choose a Mode", ["Sentiment & Emotion Analysis", "AI Engagement Chatbot"])

if mode == "Sentiment & Emotion Analysis":
    st.subheader("Analyze Sentiment & Emotion of Posts")
    user_input = st.text_area("Enter social media posts (one per line):")

    if st.button("Analyze"):
        if user_input.strip():
            posts = user_input.split("\n")
            results = []
            sentiments = []
            emotions = []

            for post in posts:
                if post.strip():
                    sentiment = sentiment_model(post)[0]
                    emotion = emotion_pipeline(post)[0]
                    polarity = TextBlob(post).sentiment.polarity

                    results.append({
                        "text": post,
                        "sentiment": sentiment['label'],
                        "sentiment_conf": sentiment['score'],
                        "emotion": emotion['label'],
                        "polarity": polarity
                    })
                    sentiments.append(polarity)
                    emotions.append(emotion['label'])

            for r in results:
                st.write(f"**Post:** {r['text']}")
                st.write(f"Sentiment: {r['sentiment']} (Confidence: {r['sentiment_conf']:.2f}) | Polarity: {r['polarity']:.2f}")
                st.write(f"Emotion: {r['emotion']}")
                st.write("---")

            st.subheader("📈 Sentiment Trend")
            plt.figure(figsize=(6, 3))
            plt.plot(sentiments, marker='o')
            plt.axhline(0, color='gray', linestyle='--')
            plt.title("Sentiment Polarity Trend")
            plt.xlabel("Post Index")
            plt.ylabel("Polarity (-1 to 1)")
            st.pyplot(plt)

            st.subheader("🎭 Emotion Distribution")
            plt.figure(figsize=(6, 3))
            emotion_counts = {e: emotions.count(e) for e in set(emotions)}
            plt.bar(emotion_counts.keys(), emotion_counts.values())
            plt.title("Emotion Frequency")
            st.pyplot(plt)
        else:
            st.warning("Please enter at least one post.")

elif mode == "AI Engagement Chatbot":
    st.subheader("Chat with Emotion-Aware AI")
    st.markdown("Persona: Friendly, empathetic assistant.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_message = st.text_input("You:", "")

    if st.button("Send") and user_message.strip():
        emotion = emotion_pipeline(user_message)[0]['label']

        if emotion == "joy":
            response = f"I'm so glad to hear that! 😊 You sound happy. Want to tell me more?"
        elif emotion == "anger":
            response = f"I understand your frustration 😠. Let's talk it through together."
        elif emotion == "sadness":
            response = f"I'm here for you 😔. It sounds like you're feeling down. Want to share?"
        elif emotion == "fear":
            response = f"I hear that you're feeling worried 😟. Let's figure out a solution together."
        else:
            response = f"Thanks for sharing! I'm here to help however I can."

        st.session_state.chat_history.append(("You", user_message))
        st.session_state.chat_history.append(("AI", response))

    for speaker, text in st.session_state.chat_history:
        st.write(f"**{speaker}:** {text}")


