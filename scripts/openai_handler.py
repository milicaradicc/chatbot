import json
from typing import List, Optional, Dict, Any
from openai import AzureOpenAI
from csv_data_fetcher import CSVDataFetcher


class OpenAIHandler:
    def __init__(self, azure_endpoint: str, azure_api_key: str, api_version: str, calendar_csv_path: str):
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=api_version
        )
        self.csv_data_fetcher = CSVDataFetcher(calendar_csv_path)
        self.tools = self._initialize_tools()

    def _initialize_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "interact_with_meeting_db",
                    "description": "Retrieves comprehensive details about calendar events and meetings. This function processes calendar entries to provide detailed information about Subject,Start Date,Start Time,End Date,End Time,All day event,Reminder on/off,Reminder Date,Reminder Time,Meeting Organizer,Required Attendees,Optional Attendees,Meeting Resources,Categories,Description,Location,Mileage,Priority. It handles various event types such as team meetings, training sessions, reviews, and workshops. The function supports both virtual and physical locations, tracks meeting resources, and manages attendee lists. Use this function when you need to retrieve event information, check scheduling conflicts, or verify meeting details. It's particularly useful for responding to queries about upcoming meetings, available resources, or participant availability. Call this when user asks 'Who is meeting organizer','Where is','When is'",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string", "description": "The subject/title of the calendar event"},
                            "start_date": {"type": "string", "description": "Start date of the event (YYYY-MM-DD format)"},
                            "start_time": {"type": "string", "description": "Start time of the event (HH:MM format)"},
                            "end_date": {"type": "string", "description": "End date of the event (YYYY-MM-DD format)"},
                            "end_time": {"type": "string", "description": "End time of the event (HH:MM format)"},
                            "all_day_event": {"type": "boolean", "description": "Whether this is an all-day event"},
                            "reminder": {"type": "string", "description": "Is reminder on or off"},
                            "meeting_organizer": {"type": "string", "description": "Who is meeting organizer"},
                            "required_attendees": {"type": "array", "items": {"type": "string"}, "description": "List of required attendees"},
                            "optional_attendees": {"type": "array", "items": {"type": "string"}, "description": "List of optional attendees"},
                            "meeting_resources": {"type": "string", "description": "Resources needed for the meeting"},
                            "categories": {"type": "string", "description": "Event categories"},
                            "description": {"type": "string", "description": "Detailed description of the event"},
                            "location": {"type": "string", "description": "Location of the event. Where its being held"},
                            "mileage": {"type": "string", "description": "Mileage information"},
                            "priority": {"type": "string", "description": "Priority level of the event"}
                        },
                        "strict":True
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "interact_with_wikipedia_db",
                    "description": "Fetches detailed information about technical topics such as programming laguages, movies, ai (e.g., 'What is Python?', 'Explain AI'). Use this tool for technical explanations or general knowledge queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query for Wikipedia content"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    def generate_response(self, query: str, context: List[str]) -> Optional[str]:
        try:
            context_str = "\n".join(context)
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant using provided context to answer queries. Use the available tools when appropriate to fetch specific information."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_str}\n\nQuery: {query}\n\nResponse:"
                }
            ]
            
            initial_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools
            )

            if not initial_response.choices[0].message.tool_calls:
                print("None")
                return initial_response.choices[0].message.content

            for tool_call in initial_response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                print(function_name)
                function_args = json.loads(tool_call.function.arguments)
                
                # Execute the right function based on the tool call
                if function_name == "interact_with_wikipedia_db":
                    function_response = self.interact_with_wikipedia_db(function_args["query"], context)
                elif function_name == "interact_with_meeting_db":
                    function_response = self.interact_with_meeting_db(function_args)
                
                if function_response:
                    # Add the function call and its response to the message history
                    messages.append(
                        {"role": "assistant", "content": None, "tool_calls": [tool_call]}
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": function_response}
                    )

            # Generate final response incorporating the tool results
            final_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                max_tokens=500,
                temperature=0
            )
            
            return final_response.choices[0].message.content

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return None

    def interact_with_meeting_db(self, query: Dict[str, Any]) -> str:
        df = self.csv_data_fetcher.df
        filtered_df = df.copy()

        if 'subject' in query:
            filtered_df = filtered_df[filtered_df['Subject'].str.contains(query['subject'], case=False, na=False)]

        if 'start_date' in query:
            filtered_df = filtered_df[filtered_df['Start Date'] == query['start_date']]

        if 'end_date' in query:
            filtered_df = filtered_df[filtered_df['End Date'] == query['end_date']]

        if 'start_time' in query:
            filtered_df = filtered_df[filtered_df['Start Time'] == query['start_time']]

        if 'end_time' in query:
            filtered_df = filtered_df[filtered_df['End Time'] == query['end_time']]

        if 'meeting_organizer' in query:
            filtered_df = filtered_df[filtered_df['Meeting Organizer'].str.contains(query['meeting_organizer'], case=False, na=False)]

        if 'required_attendees' in query:
            filtered_df = filtered_df[filtered_df['Required Attendees'].str.contains(query['required_attendees'], case=False, na=False)]

        if 'location' in query:
            filtered_df = filtered_df[filtered_df['Location'].str.contains(query['location'], case=False, na=False)]

        if 'priority' in query:
            filtered_df = filtered_df[filtered_df['Priority'].str.contains(query['priority'], case=False, na=False)]

        if not filtered_df.empty:
            print(filtered_df.to_string(index=False))
            return filtered_df.to_string(index=False)
        else:
            return "No matching events found."

    def interact_with_wikipedia_db(self, query: str, context: List[str]) -> Optional[str]:
        context_str = "\n".join(context)
        prompt = f"Context:\n{context_str}\n\nQuery: {query}\n\nResponse:"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a helpful assistant using provided context to answer queries."},
                          {"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            return None
