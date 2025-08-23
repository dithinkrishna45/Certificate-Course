emoji = {"sad":"🥲 ","happy":"😂 ","smile":"😁 ","angry":"😡 "}

text = input("Enter a Sentence :")
# text=text.lower()

for i,j in emoji.items():
    text=text.lower().replace(i,j)

print(text)
