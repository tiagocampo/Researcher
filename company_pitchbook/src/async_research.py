from typing import Dict, List, Any, Optional, Set, Tuple, cast
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
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
from bs4 import BeautifulSoup
import json
from src.models import (
    WebPage,
    CompanyAnalysis,
    CompanyInfo,
    LocationInfo,
    ProductServiceInfo,
    BusinessModelInfo
)
from src.web_navigator import WebNavigator
from src.utils.html_parser import HTMLParser
import os

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
    context: Dict[str, Any] = Field(default_factory=dict)
    
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
    completed: bool = Field(default=False, description="Whether the research workflow is completed")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def add_error(self, error: str, task_id: Optional[str] = None) -> None:
        """Add an error to the state"""
        self.errors.append({
            "error": error,
            "task_id": task_id,
            "timestamp": datetime.now()
        })
    
    def mark_task_completed(self, task_id: str) -> None:
        """Mark a task as completed"""
        if task_id not in self.completed_tasks:
            self.completed_tasks.append(task_id)
        
        # Check if all tasks are completed
        all_completed = all(
            task.status == "completed"
            for task in self.tasks.values()
        )
        if all_completed:
            self.completed = True

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
        
        # Execute task with research context
        result = agent.invoke({
            "input": task.prompt,
            "context": {
                "task_id": task_id,
                "task_name": task.name,
                "task_description": task.description,
                **state.research_context
            }
        })
        
        # Extract and format the result
        output = result.get("output", "")
        if isinstance(output, str):
            try:
                # Try to parse as JSON if it's a string
                if "```json" in output:
                    json_str = output.split("```json")[1].split("```")[0].strip()
                    output = json.loads(json_str)
                elif "```" in output:
                    json_str = output.split("```")[1].strip()
                    output = json.loads(json_str)
                else:
                    # Try to parse the entire string as JSON
                    output = json.loads(output)
            except json.JSONDecodeError:
                # If not JSON, keep as string
                pass
        
        # Format the result data
        formatted_result = {"output": {}}
        
        if isinstance(output, dict):
            # Format location data
            if "location" in output:
                location_data = output["location"]
                if isinstance(location_data, dict):
                    location_str = []
                    if location_data.get("headquarters"):
                        location_str.append(f"Headquarters: {location_data['headquarters']}")
                    if location_data.get("offices"):
                        offices = location_data["offices"]
                        if isinstance(offices, list) and offices:
                            location_str.append(f"Offices: {', '.join(offices)}")
                    if location_data.get("regions"):
                        regions = location_data["regions"]
                        if isinstance(regions, list) and regions:
                            location_str.append(f"Regions: {', '.join(regions)}")
                    formatted_result["output"]["location"] = "\n".join(location_str)
            
            # Format business model data
            if "business_model" in output:
                bm_data = output["business_model"]
                if isinstance(bm_data, dict):
                    bm_str = []
                    if bm_data.get("type"):
                        bm_str.append(f"Type: {bm_data['type']}")
                    if bm_data.get("revenue_model"):
                        bm_str.append(f"Revenue Model:\n{bm_data['revenue_model']}")
                    if bm_data.get("market_position"):
                        bm_str.append(f"Market Position:\n{bm_data['market_position']}")
                    if bm_data.get("target_industries"):
                        industries = bm_data["target_industries"]
                        if isinstance(industries, list) and industries:
                            bm_str.append("Target Industries:")
                            for ind in industries:
                                bm_str.append(f"- {ind}")
                    if bm_data.get("competitive_advantages"):
                        advantages = bm_data["competitive_advantages"]
                        if isinstance(advantages, list) and advantages:
                            bm_str.append("\nCompetitive Advantages:")
                            for adv in advantages:
                                bm_str.append(f"- {adv}")
                    formatted_result["output"]["business_model"] = "\n".join(bm_str)
            
            # Format products/services data
            if "products_services" in output:
                ps_data = output["products_services"]
                if isinstance(ps_data, dict):
                    ps_str = []
                    if ps_data.get("main_offerings"):
                        offerings = ps_data["main_offerings"]
                        if isinstance(offerings, list) and offerings:
                            ps_str.append("Main Offerings:")
                            for offering in offerings:
                                ps_str.append(f"- {offering}")
                    if ps_data.get("categories"):
                        categories = ps_data["categories"]
                        if isinstance(categories, list) and categories:
                            ps_str.append("\nCategories:")
                            for cat in categories:
                                ps_str.append(f"- {cat}")
                    if ps_data.get("descriptions"):
                        descriptions = ps_data["descriptions"]
                        if isinstance(descriptions, list) and descriptions:
                            ps_str.append("\nDescriptions:")
                            for desc in descriptions:
                                ps_str.append(f"- {desc}")
                    if ps_data.get("features"):
                        features = ps_data["features"]
                        if isinstance(features, list) and features:
                            ps_str.append("\nFeatures:")
                            for feature in features:
                                ps_str.append(f"- {feature}")
                    if ps_data.get("target_markets"):
                        markets = ps_data["target_markets"]
                        if isinstance(markets, list) and markets:
                            ps_str.append("\nTarget Markets:")
                            for market in markets:
                                ps_str.append(f"- {market}")
                    formatted_result["output"]["products_services"] = "\n".join(ps_str)
            
            # Copy any other fields
            for key, value in output.items():
                if key not in formatted_result["output"]:
                    formatted_result["output"][key] = value
        else:
            # If output is not a dict, store it as is
            formatted_result["output"] = output
        
        # Update task result
        task.status = "completed"
        task.end_time = datetime.now()
        task.result = formatted_result
        
        # Update state
        state.messages.extend([
            HumanMessage(content=task.prompt),
            AIMessage(content=str(formatted_result["output"]))
        ])
        state.tasks[task_id] = task
        state.mark_task_completed(task_id)
        state.current_task = None
        state.extracted_data[task_id] = formatted_result.get("output", {})
        
        return state
        
    except Exception as e:
        # Handle task failure
        logging.error(f"Error executing task {task_id}: {str(e)}")
        task.status = "failed"
        task.error = str(e)
        task.retry_count += 1
        
        state.add_error(str(e), task_id)
        
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
        llm: Optional[ChatOpenAI] = None,
        tools: Optional[List[Tool]] = None,
        max_retries: int = 3,
        max_tokens: int = 100000
    ):
        # Initialize LLM if not provided
        if not llm:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("No API key provided and OPENAI_API_KEY not found in environment")
            llm = ChatOpenAI(
                api_key=api_key,
                temperature=0,
                model="gpt-4"
            )
        
        self.llm: ChatOpenAI = llm
        self.tools: List[Tool] = tools or []
        self.max_retries: int = max_retries
        self.max_tokens: int = max_tokens
        self.tool_executor: ToolExecutor = ToolExecutor(self.tools)
        self.state: ResearchState = ResearchState()
        
        # Create workflow graph
        self.workflow: Graph = self._create_workflow()
    
    def _create_workflow(self) -> Graph:
        """Create the research workflow graph"""
        # Initialize graph with state type
        workflow = StateGraph(ResearchState)
        
        # Create agent
        agent = create_research_agent(
            llm=self.llm,
            tools=self.tools,
            system_prompt="""You are an expert research agent.
Your task is to analyze information and extract relevant insights.
Follow the research context and focus on providing accurate and valuable information."""
        )
        
        # Add task execution node with task_id from state
        workflow.add_node(
            "execute_task",
            lambda state: task_executor_node(state, agent=agent, task_id=state.current_task)
        )
        
        # Add task selection node
        workflow.add_node(
            "select_next_task",
            lambda state: select_next_task(state)
        )
        
        # Add end node that marks the workflow as complete
        workflow.add_node("end", lambda state: end_workflow(state))
        
        # Add conditional edges with proper stop conditions
        workflow.add_conditional_edges(
            "execute_task",
            lambda x: "end" if x.completed else "retry" if should_retry_task(x) else "select_next_task",
            {
                "retry": "execute_task",
                "select_next_task": "select_next_task",
                "end": "end"
            }
        )
        
        # Add edges from task selection to either execution or end
        workflow.add_conditional_edges(
            "select_next_task",
            lambda x: "end" if not x.current_task else "execute_task",
            {
                "execute_task": "execute_task",
                "end": "end"
            }
        )
        
        # Set entry point
        workflow.set_entry_point("select_next_task")
        
        # Compile workflow
        return workflow.compile()
    
    def add_task(self, task: AsyncResearchTask) -> None:
        """Add a task to the orchestrator"""
        if task.id in self.state.tasks:
            raise ValueError(f"Task {task.id} already exists")
        
        # Initialize research context if not present
        if not hasattr(self.state, 'research_context'):
            self.state.research_context = {}
        
        # Update research context from task
        if task.context:
            self.state.research_context.update(task.context)
        
        # Ensure timestamp is present
        if 'timestamp' not in self.state.research_context:
            self.state.research_context['timestamp'] = datetime.now()
        
        self.state.tasks[task.id] = task
    
    async def execute_all(self) -> Dict[str, Any]:
        """Execute the entire research workflow"""
        try:
            # Initialize state if not already done
            if not self.state:
                self.state = ResearchState()
            
            # Ensure research context has timestamp
            if not hasattr(self.state, 'research_context') or 'timestamp' not in self.state.research_context:
                self.state.research_context = {
                    'timestamp': datetime.now(),
                    **(getattr(self.state, 'research_context', {}) or {})
                }
            
            # Convert state to dict for workflow
            state_dict = {
                "messages": self.state.messages,
                "tasks": self.state.tasks,
                "completed_tasks": self.state.completed_tasks,
                "current_task": self.state.current_task,
                "research_context": self.state.research_context,
                "extracted_data": self.state.extracted_data,
                "errors": self.state.errors,
                "completed": self.state.completed
            }
            
            # Execute workflow with recursion limit
            try:
                result_dict = await self.workflow.ainvoke(
                    state_dict,
                    {"recursion_limit": 100}  # Set recursion limit in invoke call
                )
                
                # Update state from result
                self.state.messages = result_dict.get("messages", [])
                self.state.tasks = result_dict.get("tasks", {})
                self.state.completed_tasks = result_dict.get("completed_tasks", [])
                self.state.current_task = result_dict.get("current_task")
                self.state.research_context = result_dict.get("research_context", {})
                self.state.extracted_data = result_dict.get("extracted_data", {})
                self.state.errors = result_dict.get("errors", [])
                self.state.completed = result_dict.get("completed", False)
                
            except Exception as e:
                logging.error(f"Error in workflow execution: {str(e)}")
                raise
            
            # Return final research results
            results = {}
            for task_id, task in self.state.tasks.items():
                if isinstance(task, AsyncResearchTask) and task.status == "completed" and task.result:
                    results[task_id] = task.result
            
            # Calculate execution time
            execution_time = 0.0
            if 'timestamp' in self.state.research_context:
                execution_time = (datetime.now() - self.state.research_context["timestamp"]).total_seconds()
            
            return {
                "results": results,
                "errors": [
                    error if isinstance(error, dict) else error.model_dump() 
                    for error in self.state.errors
                ],
                "execution_time": execution_time
            }
            
        except Exception as e:
            logging.error(f"Error executing research workflow: {str(e)}")
            raise
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of task execution"""
        if not hasattr(self, 'state'):
            return {
                "status": "not_started",
                "tasks": [],
                "errors": []
            }
        
        tasks_completed = all(
            isinstance(task, AsyncResearchTask) and task.status == "completed"
            for task in self.state.tasks.values()
        )
        
        return {
            "status": "completed" if tasks_completed else "in_progress",
            "tasks": [
                {
                    "id": task_id,
                    "name": task.name if isinstance(task, AsyncResearchTask) else "Unknown",
                    "status": task.status if isinstance(task, AsyncResearchTask) else "unknown",
                    "execution_time": (
                        (task.end_time - task.start_time).total_seconds()
                        if isinstance(task, AsyncResearchTask) and task.end_time and task.start_time
                        else None
                    ),
                    "error": task.error if isinstance(task, AsyncResearchTask) else None,
                    "retry_count": task.retry_count if isinstance(task, AsyncResearchTask) else 0
                }
                for task_id, task in self.state.tasks.items()
            ],
            "errors": [
                error if isinstance(error, dict) else error.model_dump() 
                for error in self.state.errors
            ],
            "total_execution_time": (
                datetime.now() - self.state.research_context["timestamp"]
            ).total_seconds() if hasattr(self.state, 'research_context') else 0
        }

    async def _analyze_page_content(self, page: WebPage) -> Dict[str, Any]:
        """Analyze page content and extract structured data"""
        try:
            # Extract structured data from HTML
            soup = BeautifulSoup(page.html, 'html5lib')
            
            # Initialize structured data
            structured_data = CompanyAnalysis(
                company_info=CompanyInfo(),
                location=LocationInfo(),
                products_services=ProductServiceInfo(),
                business_model=BusinessModelInfo()
            )
            
            # Extract schema.org metadata first using HTMLParser
            schema_data = HTMLParser.extract_schema_metadata(soup)
            if schema_data:
                self._update_structured_data(structured_data, schema_data)
            
            # Extract HTML metadata using HTMLParser
            html_data = HTMLParser.extract_html_metadata(soup)
            if html_data:
                self._update_structured_data(structured_data, html_data)
            
            # Use LLM to analyze content and fill in missing information
            content = self._clip_text_to_token_limit(page.content, self.max_tokens)
            
            # Enhanced prompt for better structured output
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at extracting structured company information from web content.
Your task is to analyze the webpage content and extract key information about the company.
You must return ONLY a valid JSON object with no additional text or formatting.

The JSON structure must follow this exact format:
{
    "company_info": {
        "name": null,
        "description": null,
        "founded": null,
        "size": null,
        "industry": null
    },
    "location": {
        "headquarters": null,
        "offices": [],
        "regions": []
    },
    "products_services": {
        "main_offerings": [],
        "categories": [],
        "descriptions": [],
        "features": [],
        "target_markets": []
    },
    "business_model": {
        "type": null,
        "revenue_model": null,
        "market_position": null,
        "target_industries": [],
        "competitive_advantages": []
    }
}

Guidelines:
- Return ONLY the JSON object, no other text
- Use null for missing values
- Use empty arrays [] for missing lists
- Keep descriptions concise
- Maintain original terminology"""),
                ("user", f"""Extract company information from this webpage:

URL: {page.url}
Title: {page.title}

Content:
{content}""")
            ])
            
            # Get LLM analysis with retries
            for attempt in range(self.max_retries):
                try:
                    response = await self.llm.ainvoke([m.to_message() for m in prompt.format_messages()])
                    
                    # Clean and parse the response
                    content = response.content.strip()
                    
                    # Remove any markdown formatting
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].strip()
                    
                    # Parse JSON
                    llm_data = json.loads(content)
                    
                    # Validate against our models
                    validated_data = CompanyAnalysis(
                        company_info=CompanyInfo(**(llm_data.get("company_info", {}) or {})),
                        location=LocationInfo(**(llm_data.get("location", {}) or {})),
                        products_services=ProductServiceInfo(**(llm_data.get("products_services", {}) or {})),
                        business_model=BusinessModelInfo(**(llm_data.get("business_model", {}) or {}))
                    )
                    
                    # Update structured data with validated LLM analysis
                    self._update_structured_data(structured_data, validated_data.model_dump())
                    break
                    
                except (json.JSONDecodeError, ValueError) as e:
                    if attempt == self.max_retries - 1:
                        logging.error(f"Failed to parse LLM response after {self.max_retries} attempts: {str(e)}")
                        logging.error(f"Response content: {response.content if 'response' in locals() else 'No response'}")
                    await asyncio.sleep(1)  # Wait before retry
            
            return structured_data.model_dump()
            
        except Exception as e:
            logging.error(f"Error analyzing page content: {str(e)}")
            return CompanyAnalysis().model_dump()
    
    def _update_structured_data(self, target: CompanyAnalysis, source: Dict[str, Any]) -> None:
        """Update structured data with new information"""
        try:
            # Update company info
            if 'company_info' in source:
                for field, value in source['company_info'].items():
                    if value and not getattr(target.company_info, field):
                        setattr(target.company_info, field, value)
            
            # Update location info
            if 'location' in source:
                if source['location'].get('headquarters') and not target.location.headquarters:
                    target.location.headquarters = source['location']['headquarters']
                
                if 'offices' in source['location']:
                    new_offices = [
                        office for office in source['location']['offices']
                        if office not in target.location.offices
                    ]
                    target.location.offices.extend(new_offices)
                
                if 'regions' in source['location']:
                    new_regions = [
                        region for region in source['location']['regions']
                        if region not in target.location.regions
                    ]
                    target.location.regions.extend(new_regions)
            
            # Update products/services info
            if 'products_services' in source:
                if not target.products_services.main_offerings and source['products_services'].get('main_offerings'):
                    target.products_services.main_offerings = source['products_services']['main_offerings']
                
                if 'categories' in source['products_services']:
                    new_categories = [
                        cat for cat in source['products_services']['categories']
                        if cat not in target.products_services.categories
                    ]
                    target.products_services.categories.extend(new_categories)
                
                if 'descriptions' in source['products_services']:
                    new_descriptions = [
                        desc for desc in source['products_services']['descriptions']
                        if desc not in target.products_services.descriptions
                    ]
                    target.products_services.descriptions.extend(new_descriptions)
            
            # Update business model info
            if 'business_model' in source:
                if source['business_model'].get('type') and not target.business_model.type:
                    target.business_model.type = source['business_model']['type']
                if source['business_model'].get('revenue_model') and not target.business_model.revenue_model:
                    target.business_model.revenue_model = source['business_model']['revenue_model']
                if source['business_model'].get('market_position') and not target.business_model.market_position:
                    target.business_model.market_position = source['business_model']['market_position']
                
                if 'target_industries' in source['business_model']:
                    new_industries = [
                        ind for ind in source['business_model']['target_industries']
                        if ind not in target.business_model.target_industries
                    ]
                    target.business_model.target_industries.extend(new_industries)
                
                if 'competitive_advantages' in source['business_model']:
                    new_advantages = [
                        adv for adv in source['business_model']['competitive_advantages']
                        if adv not in target.business_model.competitive_advantages
                    ]
                    target.business_model.competitive_advantages.extend(new_advantages)
                    
        except Exception as e:
            logging.error(f"Error updating structured data: {str(e)}")
    
    def _clip_text_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Clip text to stay within token limit while preserving meaning"""
        try:
            # Simple character-based approximation (4 chars per token)
            chars_per_token = 4
            max_chars = max_tokens * chars_per_token
            
            if len(text) <= max_chars:
                return text
            
            # Try to clip at sentence boundary
            sentences = text[:max_chars].split('.')
            if len(sentences) > 1:
                return '.'.join(sentences[:-1]) + '.'
            
            # Fallback to character limit
            return text[:max_chars]
            
        except Exception as e:
            logging.error(f"Error clipping text: {str(e)}")
            return text[:max_tokens * 4]  # Fallback to simple character limit

def select_next_task(state: ResearchState) -> ResearchState:
    """Select the next task to execute"""
    # Get all ready tasks
    ready_tasks = [
        task_id
        for task_id, task in state.tasks.items()
        if task.status == "pending" and
        all(dep in state.completed_tasks for dep in task.dependencies)
    ]
    
    if ready_tasks:
        state.current_task = ready_tasks[0]
    else:
        state.current_task = None
        state.completed = True
    
    return state

def end_workflow(state: ResearchState) -> ResearchState:
    """Mark the workflow as complete and return the final state"""
    state.completed = True
    return state