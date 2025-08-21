# Set Practice Problems
# 1. Unique Subjects
# Create a set of subjects chosen by students.
# Add a new subject.
# Remove a subject.
# Display all subjects.
# Show total number of unique subjects.
# 3. Check Membership
# Create a set of fruits.
# Ask the user to enter a fruit name.
# Check if it is present in the set or not.

subject = set()

while True:
    print ("1.Add\n2.Remove\n3.display\n4.No.of Subjects\n5.Exit")
    ch = int(input("Enter Your Choice :"))

    if ch == 1:
        sub = input("Enter the Subject :")
        subject.add(sub)

    elif ch == 2:
        sub = input("Enter the subject you want to remove :")
        if sub in subject:
            subject.remove(sub)
        else :
            print("item not found")
    
    elif ch == 3 :
        for i in subject:
            print(i)

    elif ch == 4:
        print("No of Subjects :",len(subject))

    elif ch == 5 :
        print("Exiting...")
        exit()
    
    else :
        print("Sorry Invalid Choice")