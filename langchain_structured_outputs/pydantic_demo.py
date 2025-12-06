from pydantic import BaseModel, EmailStr, Field
# EmailStr is used for validating email format
# Field is used for adding extra information/constraints to the fields in the model
from typing import Optional

class Student(BaseModel):
    name : str = 'Azaan'
    age : Optional[int] = None
    email : EmailStr
    cgpa : float = Field(gt = 0, lt = 10.1)
# This '=' is used setting default value for the field

new_student = {'age' : 20, 'email' : 'abc@gmail.com', 'cgpa' : 10}
student = Student(**new_student)

student_dict= dict(student)
# We can convert the pydantic dictionary to python dictionary

student_json = student.model_dump_json()
# We can convert the pydantic model to json format
print(student_dict)
print(student_json)