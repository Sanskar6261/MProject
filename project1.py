import os as os
if __name__ == "__main__":
   print("HELLO WELCOME")
   while True:
       value =input("Enter your words :")
       if value=="q":
            print("Bye Bye sir")
            break
       command=f"say {value}"
       os.system(command)