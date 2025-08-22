groccery = []

while True:
    print("1.Add\n2.View\n3.Exit")
    ch = int(input("Enter Your Choice :"))

    if ch == 1:
        name = input("Enter the name of product :")
        price = float(input("Enter Price of the Product :"))
        qty = int(input("Enter the Quantity :"))

        if price > 50 :
            price = price*.95
            groccery.append({"Name":name , "Price":price,"Qty":qty})
        else:
            groccery.append({"Name":name , "Price":price,"Qty":qty})
    
    elif ch == 2:
        print(groccery)

    elif ch == 3:
        break

    else:
        print("Invalid Input!!!")