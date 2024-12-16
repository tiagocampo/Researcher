from typing import Dict, List, Any, Optional, Set, Tuple, cast
from datetime import datetime
from pydantic import BaseModel, Field
import asyncio
import logging
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph import Graph, StateGraph
from functools import partial

class AsyncResearchTask(BaseModel):
    """Represents an asynchronous research task"""
    id: str
    name: str
    description: str
    prompt: str
    dependencies: List[str] = Field(default_factory=list)
    required_tools: List[str] = Field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed, blocked
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def add_dependency(self, task_id: str):
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)
    
    def can_retry(self) -> bool:
        return self.status == "failed" and self.retry_count < self.max_retries

class TaskExecutionError(Exception):
    """Custom error for task execution failures"""
    def __init__(self, task_id: str, message: str, original_error: Optional[Exception] = None):
        self.task_id = task_id
        self.message = message
        self.original_error = original_error
        super().__init__(f"Task {task_id} failed: {message}")

class ResearchState(BaseModel):
    """State for research workflow"""
    messages: List[BaseMessage] = Field(default_factory=list)
    tasks: Dict[str, AsyncResearchTask] = Field(default_factory=dict)
    completed_tasks: List[str] = Field(default_factory=list)
    current_task: Optional[str] = None
    research_context: Dict[str, Any] = Field(default_factory=dict)
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    errors: List[Dict[str, Any]] = Field(default_factory=list)

def create_research_agent(
    llm: ChatOpenAI,
    tools: List[Tool],
    system_prompt: str
) -> AgentExecutor:
    """Create a research agent with tools"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)

def task_executor_node(
    state: ResearchState,
    agent: AgentExecutor,
    task_id: str
) -> ResearchState:
    """Execute a research task"""
    task = state.tasks[task_id]
    
    try:
        # Update task status
        task.status = "running"
        task.start_time = datetime.now()
        
        # Execute task
        result = agent.invoke({
            "messages": state.messages + [HumanMessage(content=task.prompt)],
            "research_context": state.research_context
        })
        
        # Update task result
        task.status = "completed"
        task.end_time = datetime.now()
        task.result = result
        
        # Update state
        return ResearchState(
            messages=state.messages + [
                HumanMessage(content=task.prompt),
                AIMessage(content=str(result["output"]))
            ],
            tasks={**state.tasks, task_id: task},
            completed_tasks=state.completed_tasks + [task_id],
            current_task=None,
            research_context=state.research_context,
            extracted_data={
                **state.extracted_data,
                task_id: result.get("output", {})
            }
        )
        
    except Exception as e:
        # Handle task failure
        task.status = "failed"
        task.error = str(e)
        task.retry_count += 1
        
        state.errors.append({
            "task_id": task_id,
            "error": str(e),
            "timestamp": datetime.now()
        })
        
        return state

def should_retry_task(state: ResearchState) -> bool:
    """Check if current task should be retried"""
    if not state.current_task:
        return False
    
    task = state.tasks[state.current_task]
    return task.can_retry()

def get_next_task(state: ResearchState) -> Optional[str]:
    """Get the next task to execute"""
    # Get all ready tasks
    ready_tasks = [
        task_id
        for task_id, task in state.tasks.items()
        if task.status == "pending" and
        all(dep in state.completed_tasks for dep in task.dependencies)
    ]
    
    if not ready_tasks:
        return None
    
    # Return first ready task
    return ready_tasks[0]

class AsyncResearchOrchestrator:
    """Orchestrates asynchronous research tasks"""
    
    def __init__(
        self,
        llm: ChatOpenAI,
        tools: List[Tool],
        max_retries: int = 3
    ):
        self.llm = llm
        self.tools = tools
        self.max_retries = max_retries
        self.tool_executor = ToolExecutor(tools)
        
        # Create workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> Graph:
        """Create the research workflow graph"""
        workflow = StateGraph(ResearchState)
        
        # Create agent
        agent = create_research_agent(
            llm=self.llm,
            tools=self.tools,
            system_prompt="""You are an expert research agent.
Your task is to analyze information and extract relevant insights.
Follow the research context and focus on providing accurate and valuable information."""
        )
        
        # Add task execution node
        workflow.add_node(
            "execute_task",
            partial(task_executor_node, agent=agent)
        )
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "execute_task",
            lambda x: "retry" if should_retry_task(x) else "next_task",
            {
                "retry": should_retry_task,
                "next_task": lambda x: not should_retry_task(x)
            }
        )
        
        # Set entry point
        workflow.set_entry_point("execute_task")
        
        # Compile workflow
        return workflow.compile()
    
    def add_task(self, task: AsyncResearchTask) -> None:
        """Add a task to the orchestrator"""
        if task.id in self.state.tasks:
            raise ValueError(f"Task {task.id} already exists")
        
        self.state.tasks[task.id] = task
    
    async def execute_all(self) -> Dict[str, Any]:
        """Execute all research tasks"""
        try:
            # Initialize state
            state = ResearchState(
                research_context={
                    "max_retries": self.max_retries,
                    "timestamp": datetime.now()
                }
            )
            
            # Execute workflow until all tasks are completed
            while True:
                next_task = get_next_task(state)
                if not next_task:
                    break
                
                state.current_task = next_task
                state = await self.workflow.acontinue_(state)
            
            return {
                "results": {
                    task_id: task.result
                    for task_id, task in state.tasks.items()
                    if task.status == "completed"
                },
                "errors": state.errors,
                "execution_time": (datetime.now() - state.research_context["timestamp"]).total_seconds()
            }
            
        except Exception as e:
            logging.error("Error in workflow execution", exc_info=True)
            raise
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of task execution"""
        if not hasattr(self, 'state'):
            return {
                "status": "not_started",
                "tasks": [],
                "errors": []
            }
        
        return {
            "status": "completed" if all(
                task.status == "completed"
                for task in self.state.tasks.values()
            ) else "in_progress",
            "tasks": [
                {
                    "id": task_id,
                    "name": task.name,
                    "status": task.status,
                    "execution_time": (
                        (task.end_time - task.start_time).total_seconds()
                        if task.end_time and task.start_time
                        else None
                    ),
                    "error": task.error,
                    "retry_count": task.retry_count
                }
                for task_id, task in self.state.tasks.items()
            ],
            "errors": self.state.errors,
            "total_execution_time": (
                datetime.now() - self.state.research_context["timestamp"]
            ).total_seconds() if hasattr(self, 'state') else 0
        }
</rewritten_file> 