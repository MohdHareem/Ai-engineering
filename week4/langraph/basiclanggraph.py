from typing import TypedDict
from langgraph.graph import StateGraph, END

# state
class State(TypedDict):
    number:int


#nodes
def double(state:State)->dict:
    boardnum=state["number"]
    newnum=boardnum*2
    print(newnum)
    return {"number":newnum}

def finish(state:State)->dict:
    boardnum=state["number"]
    print(boardnum)
    return {"number":boardnum}

#decision is just a fn it  is not a node 


def decision(state:State)->str:
    if state["number"] < 100:
        return "double"
    else:
        return "finish"

#graph build ---
builder=StateGraph(State)
builder.add_node("double",double)
builder.add_node("finish",finish)

builder.set_entry_point("double")
builder.add_conditional_edges(
    "double",
    decision,
    {"double":"double","finish":"finish"}
)
builder.add_edge("finish",END)
graph=builder.compile()


if __name__ == "__main__" :
    result=graph.invoke({"number":5})
    print("Result:", result)