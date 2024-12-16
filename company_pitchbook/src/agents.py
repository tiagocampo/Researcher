from typing import Dict, List, Any, TypedDict, Annotated, Sequence, Optional, Set, Tuple, cast
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.prebuilt import ToolInvocation
import operator
from enum import Enum
from dataclasses import dataclass, field
from functools import partial

class ResearchSource(BaseModel):
    url: str
    title: str
    content: str
    timestamp: datetime
    relevance_score: float = Field(ge=0, le=1)
    topics: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)

class Citation(BaseModel):
    source: ResearchSource
    quote: str
    context: str
    timestamp: datetime
    topics: List[str] = Field(default_factory=list)

class ResearchRelationship(BaseModel):
    source_topic: str
    target_topic: str
    relationship_type: str
    strength: float = Field(ge=0, le=1)
    evidence: List[Citation] = Field(default_factory=list)
    discovered_at: datetime = Field(default_factory=datetime.now)

class TopicNode(BaseModel):
    name: str
    importance_score: float = Field(ge=0, le=1)
    related_topics: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    last_explored: Optional[datetime] = None

@dataclass
class ResearchGraph:
    nodes: Dict[str, TopicNode] = field(default_factory=dict)
    relationships: List[ResearchRelationship] = field(default_factory=list)
    exploration_frontier: Set[str] = field(default_factory=set)
    
    def add_topic(self, topic: str, importance: float = 0.5) -> None:
        if topic not in self.nodes:
            self.nodes[topic] = TopicNode(
                name=topic,
                importance_score=importance
            )
            self.exploration_frontier.add(topic)
    
    def add_relationship(self, relationship: ResearchRelationship) -> None:
        self.relationships.append(relationship)
        self.nodes[relationship.source_topic].related_topics.append(relationship.target_topic)
    
    def get_next_topics_to_explore(self, max_topics: int = 3) -> List[str]:
        return sorted(
            list(self.exploration_frontier),
            key=lambda t: self.nodes[t].importance_score,
            reverse=True
        )[:max_topics]
    
    def mark_topic_explored(self, topic: str) -> None:
        if topic in self.exploration_frontier:
            self.exploration_frontier.remove(topic)
            if topic in self.nodes:
                self.nodes[topic].last_explored = datetime.now()

class ResearchState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    citations: List[Citation]
    next_steps: List[str]
    completed_tasks: List[str]
    current_task: str
    research_status: Dict[str, Any]
    research_graph: ResearchGraph
    current_depth: int
    max_depth: int

class ResearchTask(str, Enum):
    COMPANY_INFO = "company_info"
    FINANCIAL_DATA = "financial_data"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    MARKET_RESEARCH = "market_research"
    NEWS_ANALYSIS = "news_analysis"
    SYNTHESIS = "synthesis"
    TOPIC_EXPLORATION = "topic_exploration"

def create_agent(
    llm: ChatOpenAI,
    tools: List[Tool],
    system_prompt: str
) -> AgentExecutor:
    """Create an agent with the given tools and system prompt"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)

def create_research_tools(
    llm: ChatOpenAI,
    research_context: Dict[str, Any]
) -> List[Tool]:
    """Create tools for research tasks"""
    search = DuckDuckGoSearchRun()
    
    return [
        Tool(
            name="search_web",
            func=search.run,
            description="Search the web for information"
        ),
        Tool(
            name="analyze_content",
            func=partial(analyze_content, llm=llm, context=research_context),
            description="Analyze content and extract relevant information"
        ),
        Tool(
            name="evaluate_source",
            func=partial(evaluate_source, llm=llm, context=research_context),
            description="Evaluate the reliability and relevance of a source"
        )
    ]

async def analyze_content(
    content: str,
    llm: ChatOpenAI,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyze content using LLM"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze the following content and extract key information.
Consider the research context and focus on relevant details.
Format the output as a JSON object with appropriate keys."""),
        ("user", f"Research Context: {context}\n\nContent: {content}")
    ])
    
    response = await llm.ainvoke([m.to_message() for m in prompt.format_messages()])
    return response.content

async def evaluate_source(
    source: Dict[str, Any],
    llm: ChatOpenAI,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate source reliability and relevance"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Evaluate the following source for:
1. Reliability
2. Relevance to research context
3. Information quality
Return a JSON object with scores and reasoning."""),
        ("user", f"Research Context: {context}\n\nSource: {source}")
    ])
    
    response = await llm.ainvoke([m.to_message() for m in prompt.format_messages()])
    return response.content

def agent_node(state: ResearchState, agent: AgentExecutor, name: str) -> ResearchState:
    """Process a state with an agent"""
    # Get the next message
    messages = state["messages"]
    
    # Get agent response
    result = agent.invoke({
        "messages": messages,
        "research_context": state["research_status"],
        "research_graph": state["research_graph"]
    })
    
    # Add agent response to messages
    new_messages = list(messages) + [
        HumanMessage(content=str(state["current_task"])),
        AIMessage(content=str(result["output"]))
    ]
    
    # Update state
    return {
        **state,
        "messages": new_messages,
        "completed_tasks": state["completed_tasks"] + [name]
    }

def create_research_graph(
    llm: ChatOpenAI,
    research_context: Dict[str, Any]
) -> Graph:
    """Create a research workflow graph"""
    # Create tools
    tools = create_research_tools(llm, research_context)
    tool_executor = ToolExecutor(tools)
    
    # Create agents for different tasks
    agents = {
        task: create_agent(
            llm=llm,
            tools=tools,
            system_prompt=f"You are an expert at {task.value} research."
        )
        for task in ResearchTask
    }
    
    # Create workflow graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes for each research task
    for task in ResearchTask:
        workflow.add_node(
            task.value,
            partial(agent_node, agent=agents[task], name=task.value)
        )
    
    # Define conditional routing
    def should_explore_further(state: ResearchState) -> bool:
        return (
            state["current_depth"] < state["max_depth"] and
            len(state["research_graph"].exploration_frontier) > 0
        )
    
    def get_next_task(state: ResearchState) -> str:
        if should_explore_further(state):
            return ResearchTask.TOPIC_EXPLORATION.value
        return ResearchTask.SYNTHESIS.value
    
    # Add edges with conditional routing
    workflow.add_edge(ResearchTask.COMPANY_INFO.value, ResearchTask.FINANCIAL_DATA.value)
    workflow.add_edge(ResearchTask.FINANCIAL_DATA.value, ResearchTask.COMPETITOR_ANALYSIS.value)
    workflow.add_edge(ResearchTask.COMPETITOR_ANALYSIS.value, ResearchTask.MARKET_RESEARCH.value)
    workflow.add_edge(ResearchTask.MARKET_RESEARCH.value, ResearchTask.NEWS_ANALYSIS.value)
    workflow.add_edge(ResearchTask.NEWS_ANALYSIS.value, ResearchTask.TOPIC_EXPLORATION.value)
    
    # Add dynamic routing for topic exploration
    workflow.add_conditional_edges(
        ResearchTask.TOPIC_EXPLORATION.value,
        get_next_task,
        {
            ResearchTask.TOPIC_EXPLORATION.value: should_explore_further,
            ResearchTask.SYNTHESIS.value: lambda x: not should_explore_further(x)
        }
    )
    
    # Set entry and exit points
    workflow.set_entry_point(ResearchTask.COMPANY_INFO.value)
    workflow.set_finish_point(ResearchTask.SYNTHESIS.value)
    
    return workflow.compile() 