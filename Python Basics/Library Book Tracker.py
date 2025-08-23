#Tuple Project – Library Book Tracker
# Problem:
# Use tuples to represent books in a library.
# Each book = (BookID, Title, Author, Year)
# Store multiple tuples in a list.
# Allow user to:
# Search book by ID or Title
# Display all books published before a given year
# Count how many books each author has

books = [(101,"Pride and Prejudice","Jane Austen",1813),(102,"The Red and the Black","Stendhal",1830),(103,"Le Père Goriot","Honoré de Balzac",1835)]

while True:
    print("1.Add\n2.Search book by ID\n3.Display all books published before a given year\n4.Count how many books each author has\n5.exit")
    ch = int(input("Enter Choice :"))
    if ch == 1 :
        Book_id = int(input("Enter Book ID :"))
        Title = input("Enter Title of Book :")
        Author = input("Enter Author :")
        Year = int(input("Enter Year :"))
        new_book = (Book_id,Title,Author,Year)
        books.append(new_book)

        print(books)
    
    elif ch == 2 :
        Book_id = int(input("Enter Book ID :"))
        found = False
        for i in books:
            if Book_id in i:
                print(i)
                found=True
        if found == False :
            print("Book Not Fount")

    elif ch == 3 :
        year = int(input("Enter the Year :"))
        found = False
        for i in books:
            if i[3]<year:
                print(i)
                found=True
        if found == False :
            print("Book Not Fount")

    elif ch == 4 :
        author = input("Enter Author name :")
        count = 0
        for i in books:
            if i[2] == author:
                count+=1

        if count == 0:
            print("Not Found")
        else:
            print("Found ",count," book")
    

    elif ch == 5:
        print("Exiting..")
        break



    else :
        print("Invalid Input")       