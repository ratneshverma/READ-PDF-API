from pydantic import BaseModel, Field


class Client(BaseModel):
    company_name: str = Field(description="What is the company name in given document")
    short_name : str = Field(description="What is the short name in given document")
    website: str = Field(description="What is the website in given document")
    address: str = Field(description="What is the address in given document")
    tier: str = Field(description="What is the tier in given document")
    phone: str = Field(description="What is the phone number in given document")
    office_phone: str = Field(description="What is the office phone number in given document")
    city_id: str = Field(description="What is the city name in given document")
    state_id: str = Field(description="What is the state name in given document")
    country_id: str = Field(description="What is the country name in given document")
    postal_code: str = Field(description="What is the postal code in given document")
    sponsor_identification: str = Field(description="What is the sponsor identification in given document")
