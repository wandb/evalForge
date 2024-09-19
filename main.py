from fasthtml.common import *
import json
from api_client import WeaveAPIClient
import os
import dotenv
import asyncio
import random
from fasthtml.common import signal_shutdown, sse_message, EventStream
from evalforge.evalforge import EvalForge
import weave

dotenv.load_dotenv()

# Initialize the WeaveAPIClient
weave_client = WeaveAPIClient()

basic_auth = base64.b64encode(f"{os.getenv('WANDB_USERNAME')}:{os.getenv('WANDB_API_KEY')}".encode()).decode()

# Set up the app headers, including daisyui and tailwind for the chat component
tlink = Script(src="https://cdn.tailwindcss.com?plugins=typography")
jtlink = Script(src="https://cdn.jsdelivr.net/gh/williamtroup/JsonTree.js@2.9.0/dist/jsontree.min.js")
mainjs = Script(open('main.js').read(), type='text/javascript')
dlink = Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css")
jtcss = Link(rel="stylesheet", href="https://cdn.jsdelivr.net/gh/williamtroup/JsonTree.js@2.9.0/dist/jsontree.js.css")
maincss = Style(open('main.css').read(), type='text/css')
sse = Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js")

headers = (tlink, dlink, jtlink, jtcss, picolink, MarkdownJS(), HighlightJS(), mainjs, maincss, sse)

# Define a global variable for total items length
total_items_length = 0


#this render function must be defined before the app is created 
def render(Item):
    try:
        messages = json.loads(Item.inputs)
    except:
        messages = Item.inputs
    try:
        outputs = json.loads(Item.output)
    except:
        outputs = Item.output
    
    # get the annotattion from the client 
    annotations = weave_client.get_feedback_for_call(Item.project_id, Item.trace_id)

    card_header = Div(
        Div(
            H3(
                Span(f"Sample {Item.id} out of {total_items_length}" if total_items_length else "No samples in DB"),
                A("(weave link)", href=f"https://wandb.ai/{Item.project_id}/weave/calls/{Item.trace_id}", target="_blank", cls="link text-blue-500 text-xs"), 
                cls="text-base font-semibold leading-6 text-gray-9000"),
            Div(
                A(
                    Span("←", cls="sr-only"),
                    Span("←", cls="h-5 w-5", aria_hidden="true"),
                    hx_get=f"/annotate/{Item.id - 2}" if Item.id > 0 else "#",
                    hx_swap="outerHTML",
                    
                    cls="relative inline-flex items-center rounded-l-md bg-cyan-500 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-cyan-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-500" + (" pointer-events-none opacity-50" if Item.id == 1 else "")
                ),
                A(
                    Span("→", cls="sr-only"),
                    Span("→", cls="h-5 w-5", aria_hidden="true"),
                    hx_get=f"/annotate/{Item.id}" if Item.id < total_items_length - 1 else "#",
                    hx_swap="outerHTML",
                    cls="relative -ml-px inline-flex items-center rounded-r-md bg-cyan-500 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-cyan-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-500" + (" pointer-events-none opacity-50" if Item.id == total_items_length - 1 else "")
                ),
                cls="flex-shrink-0"
            ),
            cls="flex justify-between items-center mx-2"
        ), cls=" w-full"
    )
    inputs_div = Div(
            H3("Inputs", cls="text-base font-semibold leading-6 text-gray-9000 p-4"),
            Div(
                Div(messages, 
                    cls="mt-1 text-sm text-gray-500 max-h-16 overflow-y-auto whitespace-pre-wrap w-full",
                    data_jsontree_js=f"""{{ 
                        'showCounts': false, 
                        'showStringQuotes': false,
                        'showArrayItemsAsSeparateObjects': true,
                        'title': {{
                            showTreeControls: false,
                            
                            'text': 'Inputs'
                        }},
                        'data': {Item.inputs} 
                    }}""",
                    id="tree-1"
                    ),
                cls="m-2"
            ),
            cls="w-1/2 shrink-0 bg-white rounded-lg mr-2 overflow-y-auto"
        )
        
    outputs_div = Div(
            H3("Output", cls="text-base font-semibold leading-6 text-gray-9000 p-4"),
            Div(
                Div(outputs, 
                    cls="mt-1 text-sm text-gray-500 whitespace-pre-wrap w-half",
                    data_jsontree_js=f"""{{ 
                        'showCounts': false, 
                        'showStringQuotes': false,
                        'showArrayItemsAsSeparateObjects': true,
                        'title': {{
                            showTreeControls: false,
                            
                            'text': 'Output'
                        }},
                        'data': {outputs} 
                    }}""",
                    id="tree-2"
                    ),
                id="main_text",
                cls="m-2 rounded-t-lg text-sm whitespace-pre-wrap"
            ),
            cls="w-1/2 shrink-0 bg-white rounded-lg"
    )
    card_buttons_form = Div(
        Form(
            Input(
                type="text",
                name="notes",
                value=annotations.get('feedback_note', ''),
                placeholder="Additional notes?",
                cls="flex-grow p-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-transparent"
            ),
            Div(
                Button("Correct",  
                       name="feedback", 
                       value="correct", 
                       cls=f"btn btn-success mr-2 hover:text-white {'' if annotations.get('has_thumbsup') else 'btn-outline'}"
                       ),
                Button("Incorrect", 
                       name="feedback", 
                       value="incorrect", 
                        cls=f"btn btn-error hover:text-white {'' if annotations.get('has_thumbsdown') else 'btn-outline'}"
                       ),
                cls="flex-shrink-0 ml-4"
            ),
            cls="flex grow max-w-xl",
            method="post",
            hx_post=f"/feedback/{Item.trace_id}", target_id=f"item_{Item.trace_id}", hx_swap="outerHTML", hx_encoding="multipart/form-data"
            
        ),
        Div(
            
            A("Run EvalGen", 
                   cls="btn  bg-cyan-500 hover:bg-cyan-600 outline outline-cyan-500 border-0", 
                   href="/run_evalgen" 
            ),
            Div("finished annotating? run EvalGen!"),
            cls="shrink ml-4 justify-end"
        ),
        cls="mt-4  w-full flex flex-row justify-between"
    )
    
    # Card component
    card = Div(
        card_header,
        Div(    
            inputs_div,
            outputs_div,
            cls="flex flex-row w-full flex-grow shrink-1 overflow-hidden  my-4"
        ),
        card_buttons_form,
        cls="flex flex-col h-full flex-grow overflow-hidden shrink-0 overflow-y-auto",
        id=f"item_{Item.trace_id}",
        style="min-height: calc(100vh - 6rem); max-height: calc(100vh - 16rem);"
    )
    return card

app, rt, texts_db, Item = fast_app('texts.db',
                                   hdrs=headers, 
                                   live=True, 
                                   render=render, 
                                   bodykw={"data-theme":"light"},
                                   id=int, 
                                   trace_id=str, 
                                   inputs=str, 
                                   output=str, 
                                   feedback=str, 
                                   project_id=str, 
                                   annotation_task=str,
                                   additional_metrics=str,
                                   annotation_type=str, 
                                   pk='trace_id' )


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


@rt("/feedback/{trace_id}", methods=['post'])
def post(trace_id: str, feedback: str = None, notes: str = None):
    print(f"Posting feedback: {feedback} and notes: {notes} for item {trace_id}")
    items = texts_db()
    item = texts_db.get(trace_id)
    feedback_type = item.annotation_type

    weave_feedback = []
    
    # but make sure to make it look nice for the rest of weave
    if feedback == 'correct':
        weave_feedback.append({"feedback_type": "wandb.reaction.1", "payload": {"emoji": "👍"}})
    elif feedback == 'incorrect':
        weave_feedback.append({"feedback_type": "wandb.reaction.1", "payload": {"emoji": "👎"}})
    
    #if a specific feedback type was provided, add it to the specific weave feedback
    if feedback_type != '':
        weave_feedback.append({"feedback_type": feedback_type, "payload": {"value": feedback, "notes": notes}})
    elif notes:
        weave_feedback.append({"feedback_type": "wandb.note.1", "payload": {"note": notes}})
            
    
    item.feedback = weave_feedback
    texts_db.update(item)
    

    # post feedback to weave api
    weave_client.post_feedback(item.project_id, item.trace_id, weave_feedback)
    
    # find the next item using list comprehension
    next_item = next((i for i in items if i.id > item.id), items[0])
    # next_item = items[idx + 1] if idx < len(items) - 1 else items[0]
    
    print(f"Updated item {item.id} with feedback: {feedback} and notes: {notes} moving to {next_item.trace_id}")
    return next_item

@rt("/", methods=['get'])
def get():
    return Main(
        H1("Welcome to Weave LLM Judge Improver", cls="text-2xl font-bold text-center text-amber-500 mb-4"),
        Form(
            H2("Enter your Weave project ID to begin", cls="text-lg text-center text-gray-600 mb-8"),
            Input(name="project_id", type="text", value="wandb/weave-evalgen-simprod", placeholder="Enter project ID", cls="w-full p-2 mb-4 border rounded"),
            Input(type="submit", value="Submit", cls="w-full p-2 bg-cyan-500 text-white rounded cursor-pointer hover:bg-cyan-600"),
            hx_get="/get_count",
            cls="w-full max-w-md mx-auto"
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
                Label("* What is your annotation task? Provide in natural language a short description to help the LLM judge learn from what you're doing", for_="category", cls="block text-sm font-medium text-gray-700 mb-1"),
                Textarea(name="annotation_task", id="annotation_task", placeholder="I'm looking at customer support calls and trying to judge if responses include PII", cls="w-full p-2 border rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 text-sm", rows=3),
                cls="mb-4"
            ), 
            Div(
                Label("* Write any additional details about metrics you want calculated by the LLM Judge creator", for_="category", cls="block text-sm font-medium text-gray-700 mb-1"),
                Textarea(name="additional_metrics", id="additional_metrics", placeholder="I want a specific finance metric which looks at a weighted moving average over the data which involves arrays", cls="w-full p-2 border rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 text-sm", rows=3),
                cls="mb-4"
            ), 
            Div("* Optional", cls="text-xs text-gray-500"),
            Div(
                
                Button("Start annotating", type="submit", cls="btn btn-primary bg-cyan-500 hover:bg-cyan-600"),
                cls="flex justify-end mt-6"
            ),
            hx_post="/start-annotation",
            hx_target="body",
            cls="bg-white shadow-md rounded-lg px-8 pt-6 pb-8 mb-4"
        ),
        id="count-form",
        cls="w-full max-w-lg mx-auto"
    )

@app.route("/start-annotation", methods=['post', 'put'])
def post(start: int, end: int, project_id: str, annotation_task: str = None, additional_metrics: str = None):
    print(f"Starting annotation from {start} to {end} for project {project_id}")

    # Fetch calls from the Weave API
    calls = weave_client.get_calls(project_id, int(start), int(end))
    all_items = texts_db()
    if len(all_items) > 0:
        import os 
        from datetime import datetime
        # move the existing database to a backup
        os.rename('./texts.db', f'./backups/{project_id}_samples_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.db')
        # Delete the existing database
        os.remove('./texts.db-shm')
        os.remove('./texts.db-wal')
    
    annotation_type = ''
    try:
        if annotation_task:
            annotation_type = weave_client.get_category_from_task(annotation_task)
    except Exception as e:
        print("could not get category from task")
        print(e)
    

    # Insert calls into the database
    for call in calls:
        feedback = call.get('feedback', None)
        ttal = len(texts_db())
        if [item for item in all_items if item.trace_id == call.get('id')]:
            print(f"Item {call.get('trace_id')} already exists")
        else:
            texts_db.insert(id=ttal+1, 
                            project_id=project_id, 
                            trace_id=call.get('id'), 
                            inputs=call.get('inputs'), 
                            output=call.get('output'), 
                            feedback=feedback,
                            annotation_task=annotation_task,
                            additional_metrics=additional_metrics,
                            annotation_type=annotation_type)
            print(f"Item {call.get('trace_id')} already exists")
    
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
        cls="container w-full mx-auto flex flex-col max-h-full"
    )
    
    return Main(
        content,
        cls="bg-gray-100 m-0",
        hx_target="this"
    )

@rt("/run_evalgen", methods=['get'])
def run_evalgen():
    return Div(
        Div(
            Div(
                H2("Running LLM Improver...", cls="text-4xl font-bold tracking-tight text-gray-900 sm:text-6xl"),
                P("Started running LLM Improver, this will take a while... You can return to this page to see progress, or check the ./output folder for results", cls="mt-6 text-lg leading-8 text-gray-600"),
                cls="mx-auto max-w-2xl text-center"
            ),
            cls="px-6 py-24 sm:py-32 lg:px-8"
        ),
        Div(
            Div(
                H3("EvalGen is running, this will take a while...", cls="text-lg font-semibold mb-2 text-gray-200"),
                Pre(
                    id="console_output",
                    hx_ext="sse",
                    sse_connect="/progress-stream",
                    hx_swap="beforeend show:bottom",
                    sse_swap="message",
                    cls="bg-black text-green-400 font-mono p-4 rounded-md h-64 overflow-y-auto whitespace-pre-wrap"
                ),
                cls="w-full max-w-3xl mx-auto"
            ),
            cls="bg-gray-800 p-6 rounded-lg shadow-lg"
        ),
        cls="bg-white"
    )
@app.route('/run_evalgen', methods=['get'])
def run_evalgen():
    weave.init("evalgen_project")
    all_items = texts_db()
    data = [item.to_dict() for item in all_items]
    forger = EvalForge()
    results = forger.predict(data)
    forged_judge = results["forged_judges"]["judge"]
    weave.publish(forged_judge, name="final_judge")
    weave.finish()
    return Div(
        H3("EvalGen is running, this will take a while..."),
        hx_ext="sse", 
        sse_connect="/progress-stream", 
        hx_swap="show:bottom", 
        sse_swap="message",
        cls="w-full"
    )

async def number_generator():
    all_items = texts_db()
    for item in all_items:
        await asyncio.sleep(1)
        yield sse_message(Div(f"Processing sample {item.id} of {len(all_items)}"))
    
    yield sse_message(Div("EvalGen is done!"))


@rt("/progress-stream")
async def get(): return EventStream(number_generator())

serve()