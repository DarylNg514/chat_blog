from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate


css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://www.skema.edu/skTestimonies/ataky-steve.jpg?Width=210&Height=210">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

general_system_template = r""" 
You are a helpful assistant. Respond to the user questions only based on the provided content.
If the question is not relevant to the provided content, suggest other relevant questions the user can ask and respond to them. 
Verify your responses and respond only when you are confident they are accurate.
----
{context}
----
"""

general_user_template = "Question: ```{question}```"

# Création du `qa_prompt` à partir de plusieurs messages
messages = [
    SystemMessagePromptTemplate.from_template(general_system_template),
    HumanMessagePromptTemplate.from_template(general_user_template)
]

qa_prompt = ChatPromptTemplate.from_messages(messages)
