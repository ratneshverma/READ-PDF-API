from pydantic import BaseModel, Field


class Demographic(BaseModel):
    demographics_collection_date: str = Field(description="What is the 'Demographics Collection Date' in given document")
    birth_date: str = Field(description="What is the 'Birth Date' in given document")
    age: str = Field(description="What is the 'Age' in given document")
    sex: str = Field(description="What is the 'Sex' in given document")
    collected_ethnicity: str = Field(description="What is the 'Collected Ethnicity' in given document")
    if_mixed_ethnicity: str = Field(description="What is the 'If Mixed ethnicity' in given document")
    other_ethnicity: str = Field(description="What is the 'Other, Ethnicity' in given document")
    collected_race: str = Field(description="What is the 'Collected Race' in given document")
    multiple_race_if_any: str = Field(description="What is the 'Multiple Race, if any' in given document")
    race_other: str = Field(description="What is the 'Race Other' in given document")
    occupation: str = Field(description="What is the 'Occupation' in given document")
    
