from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel

def write_report(filename, html):
    with open(filename, 'w') as f:
        f.write(html)

class WriteReportArgsSchema(BaseModel):
    filename: str
    html: str

# StructuredTool for multiple args while Tool can just have one
wite_report_tool = StructuredTool.from_function(
    name="write_report",
    description="Write a HTML file to disk. Use this toll whenever someone ask for the report",
    func=write_report,
    args_schema=WriteReportArgsSchema
)