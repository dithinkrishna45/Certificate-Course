cart = []
while True:
    print("*****************")
    print("Shopping Cart")
    print("Select an option Below:")
    print("1.Add item\n2.Remove item\n3.Update item\n4.View Cart\n5.Search item\n6.Slice list\n7.Sort List\n8.Length of Cart\n9.Delete\n0.Exit")

    a = int(input("Enter Your Choice :"))

    if a == 1 : #Add item
        item = input("Enter the Item Name :")
        cart.append(item)
        print(f"{item} added to Cart Succesfully")

    elif a == 2: #Remove item
        item = input("Enter the Item Name :")
        if item in cart:
            cart.remove(item)
            print(f"{item} removed from Cart Succesfully")
        else :
            print(f"{item} Not Found in Cart")        

    elif a == 3 : #Update item
        item = input("Enter the Item Name you want to update :")
        if item in cart:
            newitem = input("Enter the new item :")
            i = cart.index(item)
            cart[i]=newitem
        else:
            print("Item not found in cart ")

    elif a == 4: #View cart
        print(f"The Cart Items are : ")
        for i in cart:
            print(i)

    elif a == 5: #Search item
        item = input("Enter the Item Name you want to search :")
        if item in cart:
            i=cart.index(item)
            print(f"{item} found at index {i}")
        else:
            print(f"{item} not found in cart")

    elif a == 6 : #Slice List
        print("To Perform Slice operation You want to enter Start index ,Stop index ,Step ")
        start = int(input("Enter Start index :"))
        stop = int(input("Enter Stop index :"))
        step = int(input("Enter Step :"))
        new_cart = cart(slice[start:stop:step])
        print(new_cart)

    elif a == 7: #Sort cart
        cart.sort()
        print("Cart Sorted Succesfully ")
        for i in cart:
            print(i)

    elif a == 8 : #Length of cart
        print(f"Length of Cart is {len(cart)}")

    elif a == 9 : #Clear Cart
        cart.clear()
        print("Cart Cleared Succesfully ")

    elif a == 0: #exit
        print("Exiting From Cart ,Thank You Visit Again 😊")
        exit()

    else :
        print("Invalid Input !!!")