from fasthtml.common import *
import json
from api_client import WeaveAPIClient
import os
import dotenv
dotenv.load_dotenv()

# Initialize the WeaveAPIClient
weave_client = WeaveAPIClient()

basic_auth = base64.b64encode(f"{os.getenv('WANDB_USERNAME')}:{os.getenv('WANDB_API_KEY')}".encode()).decode()

# Set up the app, including daisyui and tailwind for the chat component
tlink = Script(src="https://cdn.tailwindcss.com?plugins=typography"),
dlink = Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css")

# Define a global variable for total items length
total_items_length = 0


#this render function must be defined before the app is created 
def render(Item):
    messages = json.loads(Item.inputs)
    
    card_header = Div(
        Div(
            H3(f"Sample {Item.id} out of {total_items_length}" if total_items_length else "No samples in DB", cls="text-base font-semibold leading-6 text-gray-9000"),
            Div(
                A(
                    Span("←", cls="sr-only"),
                    Span("←", cls="h-5 w-5", aria_hidden="true"),
                    hx_get=f"/annotate/{Item.id - 2}" if Item.id > 0 else "#",
                    hx_swap="outerHTML",
                    
                    cls="relative inline-flex items-center rounded-l-md bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600" + (" pointer-events-none opacity-50" if Item.id == 1 else "")
                ),
                A(
                    Span("→", cls="sr-only"),
                    Span("→", cls="h-5 w-5", aria_hidden="true"),
                    hx_get=f"/annotate/{Item.id}" if Item.id < total_items_length - 1 else "#",
                    hx_swap="outerHTML",
                    cls="relative -ml-px inline-flex items-center rounded-r-md bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600" + (" pointer-events-none opacity-50" if Item.id == total_items_length - 1 else "")
                ),
                cls="flex-shrink-0"
            ),
            cls="flex justify-between items-center mb-4"
        ),
        Div(
            Div(
                P(messages, cls="mt-1 text-sm text-gray-500 max-h-16 overflow-y-auto whitespace-pre-wrap"),
                cls="ml-4 mt-4"
            ),
            cls="-ml-4 -mt-4 flex flex-wrap items-center justify-between sm:flex-nowrap"
        ),
        cls="border-b border-gray-200 bg-white p-4"
    )
    card_buttons_form = Div(
        Form(
            Input(
                type="text",
                name="notes",
                value=Item.feedback,
                placeholder="Additional notes?",
                cls="flex-grow p-2 my-4 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-transparent"
            ),
            Div(
                Button("Correct",  
                       name="feedback", 
                       value="correct", 
                       cls=f"btn btn-success mr-2 hover:text-white {'' if Item.feedback == 'correct' else 'btn-outline'}"
                       ),
                Button("Incorrect", 
                       name="feedback", 
                       value="incorrect", 
                        cls=f"btn btn-error hover:text-white {'' if Item.feedback == 'incorrect' else 'btn-outline'}"
                       ),
                cls="flex-shrink-0 ml-4"
            ),
            cls="flex items-center",
            method="post",
            hx_post=f"/feedback/{Item.id}", target_id=f"item_{Item.id}", hx_swap="outerHTML", hx_encoding="multipart/form-data"
            
        ),
        cls="mt-4"
    )
    
    # Card component
    card = Div(
        card_header,
        Div(
            Div(
                Item.output,
                id="main_text",
                cls="mt-2 w-full rounded-t-lg text-sm whitespace-pre-wrap h-auto marked"
            ),
            cls="bg-white shadow rounded-b-lg p-4 pt-0 pb-10 flex-grow overflow-scroll"
        ),
        card_buttons_form,
        cls="  flex flex-col h-full flex-grow overflow-auto",
        id=f"item_{Item.id}",
        style="min-height: calc(100vh - 6rem); max-height: calc(100vh - 16rem);"
    )
    return card

app, rt, texts_db, Item = fast_app('texts.db',hdrs=(tlink, dlink, picolink, MarkdownJS(), HighlightJS()), live=True, id=int, trace_id=str, inputs=str, output=str, feedback=str, pk='trace_id', render=render, bodykw={"data-theme":"light"})


title = 'EvalGen project'
total_items_length = len(texts_db())
# if total_items_length == 0:
#     print("No items found")
#     import json
#     with open('./data/dummy_data.jsonl', 'r') as file:
#         for line in file:
#             item = json.loads(line)
#             texts_db.insert(messages=json.dumps(item), feedback=None, notes='')
    
#     # Update total_items_length after inserting dummy data
#     total_items_length = len(texts_db())
#     print(f"Inserted {total_items_length} items from dummy data")


@rt("/feedback/{idx}", methods=['post'])
def post(idx: int, feedback: str = None, notes: str = None):
    print(f"Posting feedback: {feedback} and notes: {notes} for item {idx}")
    items = texts_db()
    item = texts_db.get(idx)
    
    item.feedback = feedback
    item.notes = notes
    texts_db.update(item)
    
    # find the next item using list comprehension
    next_item = next((i for i in items if i.id > item.id), items[0])
    # next_item = items[idx + 1] if idx < len(items) - 1 else items[0]
    
    print(f"Updated item {item.id} with feedback: {feedback} and notes: {notes} moving to {next_item.id}")
    return next_item

@rt("/", methods=['get'])
def get():
    return Main(
        H1("Welcome to Weave EvalGen", cls="text-2xl font-bold text-center text-gray-800 mb-4"),
        Form(
            H2("Enter your Weave project ID to begin", cls="text-lg text-center text-gray-600 mb-8"),
            Input(name="project_id", type="text", value="wandb/weave-evalgen-simprod", placeholder="Enter project ID", cls="w-full p-2 mb-4 border rounded"),
            Input(type="submit", value="Submit", cls="w-full p-2 bg-blue-500 text-white rounded cursor-pointer hover:bg-blue-600"),
            hx_get="/get_count",
            cls="w-full max-w-md mx-auto",
        ),
        cls="container mx-auto min-h-screen bg-gray-100 p-8 flex flex-col items-center justify-center"
    )

@rt("/get_count")
def get(project_id: str):
    count = weave_client.get_sample_count(project_id)
    
    return Div(
        
        Form(
            H2(f"Select the range of samples to annotate out of {count} samples", cls="text-lg font-semibold mb-4"),
            Div(
                Div(
                    Label("Start", for_="start", cls="block text-sm font-medium text-gray-700 mb-1"),
                    Input(name="start", type="number", id="start", placeholder="Start", min="0", max=str(count-1), value="0", cls="w-full p-2 border rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"),
                    cls="mb-4"
                ),
                Div(
                    Label("End", for_="end", cls="block text-sm font-medium text-gray-700 mb-1"),
                    Input(name="end", type="number", id="end", placeholder="End", min="1", max=int(count), value=min(int(count), 25), cls="w-full p-2 border rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"),
                    cls="mb-4"
                ),
                Input(type="hidden", name="project_id", value=project_id),
                cls="grid grid-cols-2 gap-4"
            ),
            Div(
                Button("Start Annotation", type="submit", cls="rounded-md bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"),
                cls="flex justify-end mt-6"
            ),
            hx_post="/start-annotation",
            hx_target="body",
            cls="bg-white shadow-md rounded-lg px-8 pt-6 pb-8 mb-4"
        ),
        id="count-form",
        cls="w-full max-w-md mx-auto"
    )

@app.route("/start-annotation", methods=['post', 'put'])
def post(start: int, end: int, project_id: str):
    print(f"Starting annotation from {start} to {end} for project {project_id}")
    
    # Fetch calls from the Weave API
    calls = weave_client.get_calls(project_id, int(start), int(end))
    
    # Insert calls into the database
    for call in calls:
        feedback = call.get('feedback', None)
        ttal = len(texts_db())
        texts_db.insert(id=ttal+1, trace_id=call.get('trace_id'), inputs=call.get('inputs'), output=call.get('output'), feedback=feedback)
    
    # Update total_items_length
    global total_items_length
    total_items_length = len(texts_db())
    
    print(f"Inserted {len(calls)} items from Weave API")
    
    # Redirect to the first item
    return RedirectResponse(f'/annotate/0', status_code=303)

@rt("/annotate/{idx}", methods=['get'])
def get(idx: int = 0):
    items = texts_db()
    
    index = idx 
    if index >= len(items):
        index = len(items) - 1 if items else 0

    # Container for card and buttons
    content = Div(
        H1(title,cls="text-xl font-bold text-center text-gray-800 mb-8"),
        items[index],
        cls="w-full max-w-2xl mx-auto flex flex-col max-h-full"
    )
    
    return Main(
        content,
        cls="container mx-auto min-h-screen bg-gray-100 p-8 flex flex-col",
        hx_target="this"
    )


serve()