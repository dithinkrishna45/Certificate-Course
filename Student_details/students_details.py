# Create a program for a Student Database using a Python dictionary.
# Your program should be able to
# 1. Add student details (roll number as key, name as value).
# 2. Update student details (change name using roll number).
# 3. Remove a student from the database.
# 4. View all students (display roll numbers and names).
# 5. Search for a student using roll number.
# 6. Show total number of students in the database.
# 7. Exit the program when the user wants.
# 👉 Use dictionary operations like:
# dict[key] = value (add/update)
# pop() (remove)
# items() (view all)
# in (search)
# len() (count)

student=dict()
while True:
    print("---------------------------")
    print("Student Database")
    print("---------------------------")
    print("1.Add Student Details")
    print("2.Update Student Details")
    print("3.Remove Student")
    print("4.View Student")
    print("5.Search Student")
    print("6.Total No.of Student")
    print("7.Exit")

    a=int(input("Enter Your Choice :"))

    if (a == 1) :
        roll_no = int(input("Enter The Roll No :"))
        name = input("Enter The Name :")
        student[roll_no]=name
        print(f"{name} added succesfully")
    
    elif a == 2 :
        chk = int(input("Enter the Roll no you want to update name :"))
        if chk in student:
            new_name = input(f"Enter the new name of the shudent with roll no {roll_no} :")
            student[chk]=new_name
        else :
            print("student not found in database !!!")

    elif a == 3:
        chk = int(input("Enter the Roll no you want to remove :"))
        if chk in student:
            student.pop(chk)
            print(f"removed succesfully")
        else:
            print("Student not found")

    elif a == 4 :
        for rollno,name in student.items():
            print(f"{rollno} : {name}")
        
    elif a == 5 :
        chk = int(input("Enter the Roll no you want to search :"))
        if chk in student:
            print(f"Student found succesfully :{student[chk]}")
        else:
            print("Student not found")

    elif a == 6:
        print("No of Students :",len(student))

    elif a == 7 :
        print("Thank You Visit Again😊")
        exit()
    
    else :
        print("Invalid input Try Again")