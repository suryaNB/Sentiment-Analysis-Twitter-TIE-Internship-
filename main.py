import streamlit as st
from textblob import TextBlob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Dataframe conversion function
def convert_to_df(sentiment):
    sentiment_dict = {'polarity': sentiment.polarity, 'subjectivity': sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['metric', 'value'])
    return sentiment_df

# tokens analysis
def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res > 0.1:
            pos_list.append(i)
            pos_list.append(res)

        elif res <= -0.1:
            neg_list.append(i)
            neg_list.append(res)
        else:
            neu_list.append(i)

    result = {'positives': pos_list, 'negatives': neg_list, 'neutral': neu_list}
    return result

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown(
        """
        <h1 style="text-align:center;">Sentiment Analysis NLP App</h1>
        """,
        unsafe_allow_html=True
    )
    st.image('https://d3caycb064h6u1.cloudfront.net/wp-content/uploads/2021/06/Untitled-3.png', use_column_width=True)
    st.subheader("-------------------------------------------------------------------------------")

    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter text to be analyzed:")
            submit_button = st.form_submit_button(label='Analyze')

        # layout
        col1, col2 = st.columns(2)
        if submit_button:

            with col1:
                st.info("Results")
                sentiment = TextBlob(raw_text).sentiment
                st.write(sentiment)

                # Emoji
                if sentiment.polarity > 0:
                    st.markdown("Sentiment:: Positive :smiley: ")
                elif sentiment.polarity < 0:
                    st.markdown("Sentiment:: Negative :angry: ")
                else:
                    st.markdown("Sentiment:: Neutral 😐 ")

                # Dataframe
                result_df = convert_to_df(sentiment)
                st.dataframe(result_df)

                # Visualization
                # Create a bar chart using Seaborn and Matplotlib
                plt.figure(figsize=(4, 4))
                sns.barplot(data=result_df, x='metric', y='value', palette='viridis')
                plt.xlabel('Metric')
                plt.ylabel('Value')
                plt.title('Metric vs. Value')

                # Display the chart in your Streamlit app
                st.pyplot()

            with col2:
                st.info("Token Sentiment")
                token_sentiments = analyze_token_sentiment(raw_text)
                st.write(token_sentiments)

    else:
        st.subheader("About")
        st.write("This is a sentiment analysis web app that uses vaderSentiment library to analyze text, extract the "
                 "stream of tokens and display the polarity & subjectivity visualization. ")

if __name__ == '__main__':
    main()
