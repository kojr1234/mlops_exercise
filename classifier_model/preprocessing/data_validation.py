from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError


class SpaceshipTitanicDataInputSchema(BaseModel):
    PassengerId: Optional[str]
    HomePlanet: Optional[str]
    CryoSleep: Optional[bool]
    Cabin: Optional[str]
    Destination: Optional[str]
    Age: Optional[float]
    VIP: Optional[bool]
    RoomService: Optional[float]
    FoodCourt: Optional[float]
    ShoppingMall: Optional[float]
    Spa: Optional[float]
    VRDeck: Optional[float]
    Name: Optional[str]
    # Transported: Optional[str] target variable should not be included


class MultipleSpaceshipTitanicDataInputs(BaseModel):
    # it will receive a list of dictionary
    # each element is a row of the dataset
    inputs: List[SpaceshipTitanicDataInputSchema]


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """Check model inputs for unprocessable values."""

    errors = None

    try:
        # replace nans so pydantic can read
        MultipleSpaceshipTitanicDataInputs(
            inputs=input_data.replace({np.nan, None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return input_data, errors
