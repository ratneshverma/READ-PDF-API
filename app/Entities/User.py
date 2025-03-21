from pydantic import BaseModel, Field


class User(BaseModel):
    first_name: str = Field(description="What is the first name in given document")
    middle_name: str = Field(description="What is the middle name in given document")
    last_name: str = Field(description="What is the last name in given document")
    address: str = Field(description="What is the address in given document")
    designation_id: str = Field(description="What is the designation id in given document")
    email_id: str = Field(description="What is the email id in given document")
    phone: str = Field(description="What is the phone number in given document")
    office_phone: str = Field(description="What is the office phone number in given document")
    city_id: str = Field(description="What is the city name in given document")
    state_id: str = Field(description="What is the state name in given document")
    country_id: str = Field(description="What is the country name in given document")
    postal_code: str = Field(description="What is the postal code in given document")
    management_id: str = Field(description="What is the management id in given document")
