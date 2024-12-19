import json
from typing import List, Optional, Dict, Any
from openai import AzureOpenAI


class OpenAIHandler:
    def __init__(self, azure_endpoint: str, azure_api_key: str, api_version: str):
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=api_version
        )
        self.tools = [
            {
                "type": "function",
                "function": {  
                    "name": "interact_with_calendar_db",
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
                            "meeting_organizer": {"type": "string", "description": "Name of the meeting organizer"},
                            "required_attendees": {"type": "array", "items": {"type": "string"}, "description": "List of required attendees"},
                            "optional_attendees": {"type": "array", "items": {"type": "string"}, "description": "List of optional attendees"},
                            "meeting_resources": {"type": "string", "description": "Resources needed for the meeting"},
                            "categories": {"type": "string", "description": "Event categories"},
                            "description": {"type": "string", "description": "Detailed description of the event"},
                            "location": {"type": "string", "description": "Location of the event"},
                            "mileage": {"type": "string", "description": "Mileage information"},
                            "priority": {"type": "string", "description": "Priority level of the event"}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "interact_with_wikipedia_db",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }
        ]

    def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute the specified function with the provided arguments.
        """
        if function_name == "interact_with_calendar_db":
            # Implement calendar interaction logic
            return f"Calendar interaction: {json.dumps(arguments)}"
        elif function_name == "interact_with_wikipedia_db":
            # Implement Wikipedia interaction logic
            return f"Wikipedia interaction: {arguments.get('location', '')}"
        else:
            return f"Function {function_name} not implemented"

    def generate_response(self, query: str, context: List[str]) -> Optional[str]:
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

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self.tools  
            )

            message = response.choices[0].message

            if message.tool_calls:
                function_responses = []
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    function_response = self.execute_function(function_name, function_args)
                    function_responses.append(function_response)

                    messages.extend([
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call]
                        },
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": function_response
                        }
                    ])

                final_response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )

                return final_response.choices[0].message.content
            else:
                return message.content

        except Exception as e:
            print(f"Error generating response: {e}")
            return None
