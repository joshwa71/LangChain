from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional


class PersonIntel(BaseModel):
    summary: str = Field(description="Summary of person")
    facts: List[str] = Field(description="Facts about person")
    topics_of_interest: List[str] = Field(description="Topics of interest of person")
    ice_breakers: List[str] = Field(
        description="Create ice breakers to open conversation with person"
    )

    def to_dict(self):
        return {"summary": self.summary, "facts": self.facts, "topics_of_interest": self.topics_of_interest, "ice_breakers": self.ice_breakers}


person_intel_parser = PydanticOutputParser(pydantic_object=PersonIntel)