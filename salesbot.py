import os
from typing import List, Dict
import pandas as pd
import json
import openai
import numpy as np


class RentalChatbot:
    def __init__(self, dataset_path: str):
        """
        Initialize the rental chatbot with Azure OpenAI configuration

        Args:
            dataset_path: Path to the CSV file containing rental properties data
        """
        # Configure OpenAI for Azure
        openai.api_type = "azure"
        openai.api_base = "https://techthops.openai.azure.com/"
        openai.api_version = "2023-03-15-preview"
        openai.api_key = "d8e53eae6a1a4ca196a0fdb524854558"

        self.deployment_name = "gpt-4o"
        # Use AzureOpenAI Constructor here
        self.client = openai.AzureOpenAI(
            api_key=openai.api_key,
            api_version=openai.api_version,
            azure_endpoint=openai.api_base)

        # Load dataset and convert numeric columns
        self.dataset = pd.read_csv(dataset_path)
        self._convert_numeric_columns()

    def _convert_numeric_columns(self):
        """Convert numeric columns to appropriate data types"""
        numeric_columns = [
            'BEDROOM_NUM', 'BATHROOM_NUM', 'BALCONY_NUM', 'PRICE_PER_UNIT_AREA',
            'FLOOR_NUM', 'TOTAL_FLOOR', 'MIN_PRICE', 'MAX_PRICE', 'PRICE_SQFT',
            'CARPET_SQFT', 'SUPERBUILTUP_SQFT', 'MIN_AREA_SQFT', 'MAX_AREA_SQFT',
            'PRICE', 'BUILTUP_SQFT', 'SUPER_SQFT'
        ]

        for col in numeric_columns:
            if col in self.dataset.columns:
                # Convert to numeric, coerce errors to NaN
                self.dataset[col] = pd.to_numeric(self.dataset[col], errors='coerce')
                # Fill NaN with 0 for numeric columns
                self.dataset[col] = self.dataset[col].fillna(0)

    def _safe_numeric_comparison(self, df_value, criteria_value):
        """Safely compare numeric values handling various data types"""
        try:
            # Convert both values to float for comparison
            df_val = float(df_value) if pd.notnull(df_value) else 0
            crit_val = float(criteria_value) if criteria_value is not None else 0
            return df_val, crit_val
        except (ValueError, TypeError):
            return 0, 0

    def _filter_properties(self, criteria: dict) -> pd.DataFrame:
        """
        Filter the dataset based on the search criteria.
        """
        filtered_df = self.dataset.copy()

        if 'min_price' in criteria and criteria['min_price']:
            filtered_df = filtered_df[filtered_df['PRICE'].apply(
                lambda x: self._safe_numeric_comparison(x, criteria['min_price'])[0] >=
                          self._safe_numeric_comparison(x, criteria['min_price'])[1]
            )]

        if 'max_price' in criteria and criteria['max_price']:
            filtered_df = filtered_df[filtered_df['PRICE'].apply(
                lambda x: self._safe_numeric_comparison(x, criteria['max_price'])[0] <=
                          self._safe_numeric_comparison(x, criteria['max_price'])[1]
            )]

        if 'city' in criteria and criteria['city']:
            filtered_df = filtered_df[filtered_df['CITY'].fillna('').str.contains(
                str(criteria['city']), case=False, na=False)]

        if 'locality' in criteria and criteria['locality']:
            filtered_df = filtered_df[filtered_df['LOCALITY'].fillna('').str.contains(
                str(criteria['locality']), case=False, na=False)]

        if 'bedroom_num' in criteria and criteria['bedroom_num']:
            filtered_df = filtered_df[filtered_df['BEDROOM_NUM'].apply(
                lambda x: self._safe_numeric_comparison(x, criteria['bedroom_num'])[0] ==
                          self._safe_numeric_comparison(x, criteria['bedroom_num'])[1]
            )]

        if 'bathroom_num' in criteria and criteria['bathroom_num']:
            filtered_df = filtered_df[filtered_df['BATHROOM_NUM'].apply(
                lambda x: self._safe_numeric_comparison(x, criteria['bathroom_num'])[0] ==
                          self._safe_numeric_comparison(x, criteria['bathroom_num'])[1]
            )]

        if 'property_type' in criteria and criteria['property_type']:
            filtered_df = filtered_df[filtered_df['PROPERTY_TYPE'].fillna('').str.contains(
                str(criteria['property_type']), case=False, na=False)]

        if 'furnish' in criteria and criteria['furnish']:
            filtered_df = filtered_df[filtered_df['FURNISH'].fillna('').str.contains(
                str(criteria['furnish']), case=False, na=False)]

        if 'facing' in criteria and criteria['facing']:
            filtered_df = filtered_df[filtered_df['FACING'].fillna('').str.contains(
                str(criteria['facing']), case=False, na=False)]

        if 'min_area_sqft' in criteria and criteria['min_area_sqft']:
            filtered_df = filtered_df[filtered_df['MIN_AREA_SQFT'].apply(
                lambda x: self._safe_numeric_comparison(x, criteria['min_area_sqft'])[0] >=
                          self._safe_numeric_comparison(x, criteria['min_area_sqft'])[1]
            )]

        if 'max_area_sqft' in criteria and criteria['max_area_sqft']:
            filtered_df = filtered_df[filtered_df['MAX_AREA_SQFT'].apply(
                lambda x: self._safe_numeric_comparison(x, criteria['max_area_sqft'])[0] <=
                          self._safe_numeric_comparison(x, criteria['max_area_sqft'])[1]
            )]

        if 'amenities' in criteria and criteria['amenities']:
            for amenity in criteria['amenities']:
                filtered_df = filtered_df[filtered_df['AMENITIES'].fillna('').str.contains(
                    str(amenity), case=False, na=False)]

        if 'features' in criteria and criteria['features']:
            for feature in criteria['features']:
                filtered_df = filtered_df[filtered_df['FEATURES'].fillna('').str.contains(
                    str(feature), case=False, na=False)]

        return filtered_df

    def _create_search_query(self, user_message: str) -> dict:
        """
        Convert user message to search criteria using Azure OpenAI API.
        """
        messages = [
            {"role": "system", "content": """
            Convert the user's rental query into a JSON search criteria.
            Include only the following fields if mentioned:
            - min_price (as integer)
            - max_price (as integer)
            - city (as string)
            - locality (as string)
            - bedroom_num (as integer)
            - bathroom_num (as integer)
            - property_type (as string)
            - furnish (as string: Furnished/Semifurnished/Unfurnished)
            - facing (as string)
            - min_area_sqft (as integer)
            - max_area_sqft (as integer)
            - amenities (as list of strings)
            - features (as list of strings)
            - profile (as list of strings)
            Only include fields that are explicitly mentioned in the query.
            """
             },
            {"role": "user", "content": user_message}
        ]

        response = self.client.chat.completions.create(
        model = self.deployment_name,
        messages = messages,
        temperature = 0.7
        )

        try:
            return json.loads(response.choices[0].message.content) # Changed to attribute access
        except json.JSONDecodeError:
            return {}

    def _generate_response(self, filtered_properties: pd.DataFrame,
                           original_query: str) -> str:
        """
        Generate a natural language response using Azure OpenAI API.
        """
        if len(filtered_properties) == 0:
            return "I couldn't find any properties matching your criteria. Would you like to broaden your search or try different parameters?"

        properties_list = filtered_properties[[
            'PROP_ID', 'PROPERTY_TYPE', 'CITY', 'LOCALITY', 'BEDROOM_NUM',
            'BATHROOM_NUM', 'PRICE', 'FURNISH', 'AMENITIES', 'TOP_USPS',
            'PROP_HEADING', 'SOCIETY_NAME', 'BUILDING_NAME', 'MIN_AREA_SQFT',
            'PROP_DETAILS_URL', 'profile'
        ]].head(5).to_dict('records')

        # Convert numpy int64/float64 to regular Python types for JSON serialization
        for prop in properties_list:
            for key, value in prop.items():
                if isinstance(value, (np.int64, np.float64)):
                    prop[key] = int(value) if isinstance(value, np.int64) else float(value)
                elif pd.isna(value):
                    prop[key] = None

        properties_json = json.dumps(properties_list)

        messages = [
            {"role": "system", "content": """You are a helpful real estate assistant.
            Provide clear, concise summaries of properties, highlighting key features and amenities."""},
            {"role": "user", "content": f"""
            Original query: {original_query}

            Available properties: {properties_json}

            Generate a helpful response summarizing the available properties.
            For each property include:
            1. Location (City, Locality, Society/Building name)
            2. Configuration (bedrooms, bathrooms)
            3. Area in sq ft
            4. Price
            5. Key USPs and amenities
            6. Property URL for more details
            7. Contact details with phone number


            Limit to top 5 properties.
            """}
        ]

        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            temperature=0.7
        )

        return response.choices[0].message.content

    def get_response(self, user_message: str) -> str:
        """
        Main method to process user queries and return responses.
        """
        try:
            # Convert user query to search criteria
            search_criteria = self._create_search_query(user_message)
            print(search_criteria)

            # Filter properties based on criteria
            matching_properties = self._filter_properties(search_criteria)
            print(matching_properties)

            # Generate natural language response
            response = self._generate_response(matching_properties, user_message)

            return response

        except Exception as e:
            return f"I apologize, but I encountered an error processing your request: {str(e)}"