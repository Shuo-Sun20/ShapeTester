"""Prompt templates and validators for LLM-driven code generation."""

from typing import Optional,Callable
from typing import cast
from pydantic import BaseModel
import re
import json
import inspect

# base prompt configuration model
class PromptConfig(BaseModel):
    prompt_CN: str
    prompt_EN: str
    result_extractor: Optional[Callable[[str], Optional[str]]] = None
    result_validator: Optional[Callable[[str], str]] = None
    retry_times: int = 3
    name:str = "BasePromptConfig"

    def update_validators(self, validator:Callable[[str], str]):
        self.result_validator = validator

# Helper functions for extracting and validating content
# basic extractor
def python_code_extractor(response: str) -> Optional[str]:
    """Extract Python code block from LLM response."""
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    return None

def json_content_extractor(response: str) -> Optional[str]:
    """Extract JSON content block from LLM response."""
    pattern = r"```json(.*)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    return None

# basic validators
def validate_json_content(content: str) -> str:
    """Validate if the content is valid JSON."""
    try:
        json.loads(content)
        return ""
    except Exception as e:
        return str(e)
    
# advanced validators

def validate_json_dict_content(content: str) -> str:
    """Validate if the content is valid JSON. and is a dict with string keys."""
    try:
        d = json.loads(content)
        if not (isinstance(d, dict) and all(isinstance(k, str) for k in d.keys())):
             raise ValueError("Content is not a JSON dict[str, Any].")
        return ""
    except Exception as e:
        return str(e)

def valid_python_code_with_call_func(code:str)-> str:
            """
            code:str 应该满足以下要求：
            1. 定义了函数call_func().
            2. call_func()的参数列表中应该有inputs.
            3. call_func()不应包含**kwargs或*args等不明确的参数定义。
            4. 存在example_output变量保存了调用call_func()的返回值。
            5. example_output变量的值应该是一个张量，或者一个包含多个张量的list。
            该函数对code的上述要求进行检查，如果满足要求返回"",否则返回不满足要求的原因。
            """
            sandbox = {"__builtins__": __builtins__, "__name__": "__main__",}
            try:
                exec(code, sandbox, sandbox)
                if "call_func" not in sandbox:
                    return "call_func() is not defined."
                call_func = sandbox["call_func"]
                if not callable(call_func):
                    return "call_func is not a function."
                args = inspect.getargs(call_func.__code__)  # ensure it's a proper function to access its signature
                if args.varargs is not None or args.varkw is not None:
                    return f"call_func() should not have *args or **kwargs in its parameters."
                if "inputs" not in call_func.__code__.co_varnames:
                    return "call_func() should have an 'inputs' parameter."
                if "example_output" not in sandbox:
                    return "example_output variable is not defined."
                example_output = sandbox["example_output"]
                if not (hasattr(example_output, "shape") or (isinstance(example_output, list) and all(hasattr(tensor, "shape") for tensor in example_output))):
                    return "example_output should be a tensor or a list of tensors."
            except Exception as e:
                return f"Code execution error: {e}"
            return ""

def custom_validator(code: str) -> str:
    raise NotImplementedError("Custom validator not implemented.")

CALL_FUNC_Generation_Config = PromptConfig(
    prompt_CN="""
    你是熟练掌握{frame_name}框架的一位深度学习工程师。我正在对{frame_name}Ver{version}中的{api_name}API进行测试。
    {api_name}的文档描述如下：{api_doc}。
    请根据你对{frame_name}的了解，完成下面的任务：
1. 编写一个函数call_func()调用API {api_name}。call_func()应该满足下列要求：
   .接受{api_name}的构造函数的所有参数（如果{api_name}是一个类）和调用时的参数作为其自身输入参数。
   .{api_name}的输入张量应该对应call_func()的"inputs"参数。如果{api_name}有多个输入张量，应该它们合并成一个list参数inputs传入call_func()，并在call_func()内部进行拆分以调用API。
   .如果{api_name}是一个类，基于输入的参数构造一个实例，并用实例接受输入，完成API调用，否则，利用输入参数直接完成API调用并返回输出的张量
   .call_func的参数列表中不应该有**kwargs或*args等不明确的参数定义。
   .只包含直接调用API的代码逻辑(必要的实例化和函数直接调用)，不包含其他多余的代码。
2. 构造一个合法的输入调用call_func(),并将返回值保存到"example_output"变量中。注意：输入的张量请用随机生成的张量表示，不要用keras.Input等占位符。
注意：以
```python
`````
的形式返回你编写的代码，不要返回其他的内容。
""",
    prompt_EN="""
    You are a deep learning engineer proficient in the {frame_name} framework. I am testing the {api_name} API in {frame_name} Ver{version}. 
The documentation description of {api_name} is as follows: {api_doc}.
    Based on your knowledge of {frame_name}, please complete the following tasks:
1. Write a function call_func() that calls the API {api_name}. call_func() should meet the following requirements:
    .Accept all parameters of the constructor (if {api_name} is a class) and the parameters when calling {api_name} as its own input parameters
   .The input tensor(s) of {api_name} should correspond to the "inputs" parameter of call_func(). If {api_name} has multiple input tensors, they should be combined into a list parameter inputs passed into call_func(), and then split within call_func() to call the API.
   .If {api_name} is a class, construct an instance based on the input parameters, use the instance to accept the input, complete the API call; otherwise, directly complete the API call using the input parameters
   .And return the output tensor
    .The parameter list of call_func should not have ambiguous parameter definitions such as **kwargs or *args.
   .Only include the code logic that directly calls the API (necessary instantiation and direct function calls), without other redundant code.
2. Construct a valid input to call call_func(), and save its return value to an "example_output" variable. Note: Please use randomly generated tensors to represent the input tensors, do not use placeholders such as keras.Input.
Note: Return your code in the form of
```python
<code>
```
Do not return any other content.
""",
    result_extractor=python_code_extractor,
    result_validator=valid_python_code_with_call_func,
    name = "CALL_FUNC_Generation_Config"
)   

Generate_Input_Space_Config = PromptConfig(
    prompt_CN="""
你是熟练掌握{frame_name}框架的一位深度学习工程师。我正在对{frame_name}Ver{version}中的{api_name}API进行测试。
{api_name}的文档描述如下：{api_doc}。
下面的代码中call_func()函数封装了{api_name}的调用，并且提供了call_func()的一个合法调用示例。
{code_snippet}
请根据代码和你对{frame_name}的了解，完成下面的任务：
1  以dict的形式定义一个valid_test_case变量，包含调用代码中call_func()的所有参数及对应的取值。注意：应保证可以通过call_func(**valid_test_case)成功调用API。
2. 识别call_func的参数列表中除了"inputs"之外**所有**能够影响输出张量的形状的参数。
3. 基于你对{frame_name}的了解，分析这些参数的类型，从而构造每个参数的取值空间。注意：对于离散型参数，直接罗列其所有可能取值；对于连续型参数，请对其取值空间进行离散化，确保包含边界值和至少5个典型值，这些典型值应尽可能覆盖所有合法取值场景；valid_test_case中定义的参数值应包含在对应的取值空间。
4. 以dataclass的形式，定义一个名为InputSpace的类，包含上述所有影响输出张量形状的参数及其离散化后的取值范围。注意：请确保每个参数的取值范围是一个列表，列表中包含所有离散化后的取值，且InputSpace的每个Field与对应的参数名称一致。应保证可以通过var=InputSpace()成功实例化InputSpace类。
注意：以
```python
```
的形式返回你编写的代码，包括变量valid_test_case的定义和InputSpace类的定义，这段代码应该能够成功运行。
不要返回其他的内容。
""",
    prompt_EN="""
You are a deep learning engineer proficient in the {frame_name} framework. I am testing the {api_name} API in {frame_name} Ver{version}. 
The documentation description of {api_name} is as follows: {api_doc}.
The following code snippet encapsulates the call to {api_name} in a call_func() function and provides a valid example of calling call_func().
{code_snippet}
Based on the code and your knowledge of {frame_name}, please complete the following tasks:
1. Define a variable valid_test_case in the form of a dict, containing all parameters of call_func() in the calling code and their corresponding values. Note: It should be ensured that the API can be successfully called through call_func(**valid_test_case).
2. Identify all parameters in the parameter list of call_func that can affect the shape of the output tensor, except for "inputs".
3. Based on your knowledge of {frame_name}, analyze the types of these parameters to construct the value space for each parameter. Note: For discrete parameters, directly list all possible values; for continuous parameters, discretize their value space to ensure that boundary values and at least 5 typical values are included. These typical values should cover all legal value scenarios as much as possible; the parameter values defined in valid_test_case should be included in the corresponding value space.
4. In the form of a dataclass, define a class named InputSpace, which contains **all** the parameters that affect the shape of the output tensor and their discretized value ranges. Note: Please ensure that the value range of each parameter is a list containing all discretized values, and that each Field of InputSpace corresponds to the name of the parameter. It should be ensured that the InputSpace class can be successfully instantiated through var=InputSpace().
Note: Return your code in the form of
```python
```
including the definition of the variable valid_test_case and the definition of the InputSpace class. This code should be able to run successfully.
Do not return any other content.
""",
    result_extractor=python_code_extractor,
    result_validator=custom_validator,
    name = "Generate_Input_Space_Config"
)

# Prompt configuration for conflict-parameter detection
Conflict_Param_Detect_Config = PromptConfig(
    prompt_CN="""
    你是熟练掌握{frame_name}框架的一位深度学习工程师。我正在对{frame_name}Ver{version}中的{api_name}API进行测试。
    {api_name}的文档描述如下：{api_doc}。
    下面的代码中call_func()函数封装了{api_name}的调用：
    {code_snippet}.
    我使用的测试用例 testcase = {test_case}, 以call_func(**testcase)的方式调用API时，出现了如下错误信息：{error_message}
    请你根据错误信息，分析导致错误的参数组合，并以列表的形式返回这些参数名称。
    注意：
    以
    ```json
    ```
    的形式返回参数名称列表，例如：```json["param1", "param2"]```，其中的参数名称应该全部来自call_func()的参数列表，不要返回其他的内容。
    """,
    prompt_EN="""
    You are a deep learning engineer proficient in the {frame_name} framework. I am testing the {api_name} API in {frame_name} Ver{version}.
    The documentation description of {api_name} is as follows: {api_doc}.
    The following code snippet encapsulates the call to {api_name} in a call_func() function:
    {code_snippet}.
    When I used the test case testcase = {test_case} to call the API in the form of call_func(**testcase), I encountered the following error message: {error_message}
    Based on the error message, please analyze the combination of parameters that leads to the error and return the names of these parameters in the form of a list.
    Note:
    Return the list of parameter names in the form of
    ```json
    ``` , for example: ```json["param1", "param2"]```, where the parameter names should all come from the parameter list of call_func(). Do not return any other content.
    """,
    result_extractor=json_content_extractor,
    result_validator=custom_validator,
    name = "Conflict_Param_Detect_Config"
)


def python_code_or_json_extractor(response: str) -> Optional[str]:
    """Extract Python code block or JSON content block from LLM response."""
    code = python_code_extractor(response)
    if code is not None:
        return code
    json_content = json_content_extractor(response)
    if json_content is not None:
        return json_content
    return None

def inputs_extension_validator(code):
        sandbox = {
                "__builtins__": __builtins__,
                "__name__": "__main__",
            }
        try:
            exec(code, sandbox, sandbox)
            if 'inputs_extension' not in sandbox:
                return "The code should define a variable named 'inputs_extension'"
            inputs_extension = sandbox['inputs_extension']
            if not isinstance(inputs_extension, list) or not all(isinstance(item, list) for item in inputs_extension):
                return "inputs_extension should be a list of lists"
        except Exception as e:
            return f"Code execution raised an exception: {e}"
        return ""

def always_valid(response: str) -> str:
    return ""

Inputs_Extension_Config = PromptConfig(
    prompt_CN="""
    你是熟练掌握{frame_name}框架的一位深度学习工程师。我正在对{frame_name}Ver{version}中的{api_name}API进行测试。
    {api_name}的文档描述如下：{api_doc}。
    我通过call_func()函数封装了{api_name}：{code_snippet}。
    我对其中一些参数的取值已经有了初步的设定，这些参数取值空间设置如下：{input_space}。
    call_func()函数的定义中，"inputs"参数表示了API的输入张量。
    请你基于call_func()函数的定义以及你对{frame_name}的了解，生成一段python代码，在这段代码中以list的形式定义一个‘inputs_extension’对象。
    ‘inputs_extension’包含至少五个list类型的元素，表示新的inputs取值，这些取值的类型应该与原来call_func()函数中"inputs"参数相同，但其中包含的张量的形状应该尽量多样化，以覆盖更多的测试场景。
    注意：每一个新的‘inputs'参数都应该能够与至少一组其他参数的取值组合成能够通过call_func()成功调用API的测试输入。
    请以```python
    ```
    的形式返回你编写的代码，这段代码应该能够成功运行。
    不要返回其他的内容。
    """,
    prompt_EN="""
    You are a deep learning engineer proficient in the {frame_name} framework. I am testing the {api_name} API in {frame_name} Ver{version}.
    The documentation description of {api_name} is as follows: {api_doc}.
    I have encapsulated {api_name} through the call_func() function: {code_snippet}.
    I have preliminary settings for the values of some parameters, and the value space of these parameters is set as follows: {input_space}.
    The "inputs" parameter in the definition of the call_func() function represents the input tensors of the API.
    Based on the definition of the call_func() function and your understanding of {frame_name}, please generate a piece of Python code in which a 'inputs_extension' object is defined in the form of a list.
    'inputs_extension' contains at least 5 elements of type list, which represent new values for "inputs" parameter, and the type of these values should be the same as the "inputs" parameter in the original call_func() function, but the shapes of the tensors contained in it should be as diverse as possible to cover more testing scenarios.
    Note: Each new "inputs" parameter should be able to be combined with at least one group of other parameter values to form a test input that can successfully call the API through call_func().
    Please return the code you wrote in the form of```python
```, and this code should be able to run successfully.
    Do not return any other content.
    """,
    result_extractor=python_code_extractor,
    result_validator=inputs_extension_validator,
    name = "Inputs_Extension_Config"
)

Test_Case_Complete_Config = PromptConfig(
    prompt_CN="""
    你是熟练掌握{frame_name}框架的一位深度学习工程师。我正在对{frame_name}Ver{version}中的{api_name}API进行测试。
    {api_name}的文档描述如下：{api_doc}。
    我通过call_func()函数封装了{api_name}：{code_snippet}。
    我对其中一些参数的取值已经有了初步的设定，这些参数取值空间设置如下：{input_space}。
    请你基于对{frame_name}的了解，生成一段python代码，在这段代码中以dict的形式定义一个‘test_case’对象，‘test_case’以param_name: param_value的形式定义一个测试用例。
    该‘test_case’需要包含一些预设的参数取值：
    {partial_test_case}。
    首先，请判断这些预定义的参数间是否存在冲突，如果存在，请分析导致错误的参数组合，并以
    ```json
    ```的形式给出冲突参数的名称列表，例如：```json["param1", "param2"]```。否则，请以
    ```python
    ```
    的形式生成定义‘test_case’的python代码。'test_case'应该可以通过call_func(**test_case)成功调用API。
    不要返回其他的内容。
    """,
    prompt_EN="""
    You are a deep learning engineer proficient in the {frame_name} framework. I am testing the {api_name} API in {frame_name} Ver{version}.
    The documentation description of {api_name} is as follows: {api_doc}.
    I have encapsulated {api_name} through the call_func() function: {code_snippet}.
    I have preliminary settings for the values of some parameters, and the value space of these parameters is set as follows: {input_space}.
    Based on your understanding of {frame_name}, please generate a piece of Python code in which a 'test_case' object is defined in the form of a dict, and 'test_case' defines a test case in the form of param_name: param_value.
    The 'test_case' needs to include some preset parameter values:
    {partial_test_case}.
    First, please determine if there are any conflicts between these predefined parameters. If there are, please analyze the combinations of parameters that lead to the error and provide a list of the names of the conflicting parameters in the form of
    ```json
    ``` , for example: ```json["param1", "param2"]```. 
    Otherwise, please generate the Python code that defines 'test_case' in the form of
    ```python
    ```.  
    'test_case' should be able to successfully call the API through call_func(**test_case).
    Do not return any other content.
    """,
    result_extractor=python_code_or_json_extractor,
    result_validator=custom_validator,
    name = "Test_Case_Complete_Config")


POC_Generation_KERAS_Config = PromptConfig(
    prompt_CN="""
    你是熟练掌握Keras框架的一位深度学习工程师，在对Keras框架中的一个API进行测试时发现了一个缺陷。
    首先，我将这个API的调用封装在一个函数call_func()中。
    缺陷的表现为：当使用eager张量作为输入调用call_func时获取的动态输出形状与使用相同形状的Keras.Input占位符调用call_func时获取的静态输出形状不一致。
    现在你的任务是根据测试输入构造一个POC代码来复现这个缺陷，以便开发人员能够快速定位和修复这个问题。
    请你根据以下信息生成一个POC代码。请以
    ```python
    ```
    的形式返回你的代码。不要返回其他的内容。
    请确保你的POC代码能够成功运行，并且能够复现该缺陷。
    
    API的名称：{api_name}。
    API的文档：{api_doc}。
    call_func()函数的定义：{code_snippet}。
    导致缺陷的测试输入：{test_case}。
    动态输出形状：{dynamic_shape}。(为统一形状，用None表示batch维，其他维度用具体数字表示)
    静态输出形状：{static_shape}。(为统一形状，用None表示batch维，其他维度用具体数字表示)
    """,
    prompt_EN="""
    You are a deep learning engineer proficient in the Keras framework, and while testing an API in the Keras framework, you found a defect.
    First, I encapsulated the call to this API in a function call_func().
    The defect manifests as: when calling this call_func with eager tensors as input, the dynamic output shape obtained is inconsistent with the static output shape obtained by calling call_func with Keras.Input placeholders of the same shape as input.
    Your current task is to construct a POC code based on the test input to reproduce this defect so that the developers can quickly locate and fix the problem.
    Please generate a POC code based on the following information. Please return your code in the form of
    ```python
    ```
    . Do not return any other content.
    Please ensure that your POC code can run successfully and can reproduce the defect.

    API name: {api_name}.
    API documentation: {api_doc}.
    Definition of call_func(): {code_snippet}.
    Test input that causes the defect: {test_case}.
    Dynamic output shape: {dynamic_shape}. (Use None to represent the batch dimension and specific numbers for other dimensions to unify the shape)
    Static output shape: {static_shape}. (Use None to represent the batch dimension and specific numbers for other dimensions to unify the shape)
    """,
    result_extractor=python_code_extractor,
    result_validator=always_valid,
    name = "POC_Generation_KERAS_Config"
)

POC_Generation_TensorFlow_Config = PromptConfig(
    prompt_CN="""
    你是熟练掌握TensorFlow框架的一位深度学习工程师，在对TensorFlow框架中的一个API进行测试时发现了一个缺陷。
    我首先将这个API的调用封装在一个函数call_func()中。
    缺陷的表现为：当直接调用该call_func时获取的动态输出形状与调用tf.function(call_func)时获取的静态输出形状不一致。
    现在你的任务是根据测试输入构造一个POC代码来复现这个缺陷，以便开发人员能够快速定位和修复这个问题。
    请你根据以下信息生成一个POC代码。请以
    ```python
    ```
    的形式返回你的代码。不要返回其他的内容。
    请确保你的POC代码能够成功运行，并且能够复现该缺陷。
    
    API的名称：{api_name}。
    API的文档：{api_doc}。
    call_func()函数的定义：{code_snippet}。
    导致缺陷的测试输入：{test_case}。
    动态输出形状：{dynamic_shape}。(为统一形状，用None表示batch维，其他维度用具体数字表示)
    静态输出形状：{static_shape}。(为统一形状，用None表示batch维，其他维度用具体数字表示)
    """,
    prompt_EN="""
    You are a deep learning engineer proficient in the TensorFlow framework, and while testing an API in the TensorFlow framework, you found a defect.
    First, I encapsulated the call to this API in a function call_func().
    The defect manifests as: when directly calling this call_func, the dynamic output shape obtained is inconsistent with the static output shape obtained by calling tf.function(call_func).
    Your current task is to construct a POC code based on the test input to reproduce this defect so that the developers can quickly locate and fix the problem.
    Please generate a POC code based on the following information. Please return your code in the form of
    ```python
    ```
    . Do not return any other content.
    Please ensure that your POC code can run successfully and can reproduce the defect.

    API name: {api_name}.
    API documentation: {api_doc}.
    Definition of call_func(): {code_snippet}.
    Test input that causes the defect: {test_case}.
    Dynamic output shape: {dynamic_shape}. (Use None to represent the batch dimension and specific numbers for other dimensions to unify the shape)
    Static output shape: {static_shape}. (Use None to represent the batch dimension and specific numbers for other dimensions to unify the shape)
    """,
    result_extractor=python_code_extractor,
    result_validator=always_valid,
    name = "POC_Generation_TensorFlow_Config")

POC_Generation_PyTorch_Config = PromptConfig(
    prompt_CN="""
    你是熟练掌握PyTorch框架的一位深度学习工程师，在对PyTorch框架中的一个API进行测试时发现了一个缺陷。
    我首先将这个API的调用封装在一个函数call_func()中。
    之后我直接调用该call_func时获取的动态输出形状，与调用torch.compile(call_func, dynamic=True)时获取的静态输出形状, 以及通过device='meta'获取的meta形状。
    缺陷的表现为三个形状之间存在二者不一致。
    现在你的任务是根据测试输入构造一个POC代码来复现这个缺陷，以便开发人员能够快速定位和修复这个问题。
    请你根据以下信息生成一个POC代码。请以
    ```python
    ```
    的形式返回你的代码。不要返回其他的内容。
    请确保你的POC代码能够成功运行，并且能够复现该缺陷。
    
    API的名称：{api_name}。
    API的文档：{api_doc}。
    call_func()函数的定义：{code_snippet}。
    导致缺陷的测试输入：{test_case}。
    动态输出形状：{dynamic_shape}。(为统一形状，用None表示batch维，其他维度用具体数字表示)
    静态输出形状：{static_shape}。(为统一形状，用None表示batch维，其他维度用具体数字表示)
    """,
    prompt_EN="""
    You are a deep learning engineer proficient in the PyTorch framework, and while testing an API in the PyTorch framework, you found a defect.
    First, I encapsulated the call to this API in a function call_func().
    After that, I obtained the dynamic output shape by directly calling the call_func, the static output shape obtained by calling torch.compile(call_func, dynamic=True), and the meta shape obtained by setting device='meta'.
    The defect manifests as inconsistencies among the three shapes.
    Your current task is to construct a POC code based on the test input to reproduce this defect so that the developers can quickly locate and fix the problem.
    Please generate a POC code based on the following information. Please return your code in the form of
    ```python
    ```
    . Do not return any other content.
    Please ensure that your POC code can run successfully and can reproduce the defect.

    API name: {api_name}.
    API documentation: {api_doc}.
    Definition of call_func(): {code_snippet}.
    Test input that causes the defect: {test_case}.
    Dynamic output shape: {dynamic_shape}. 
    Static output shape: {static_shape}. 
    Meta output shape: {meta_shape}. 
    """,
    result_extractor=python_code_extractor,
    result_validator=always_valid,
    name = "POC_Generation_PyTorch_Config"
)


POC_Generation_Config_Dict = {
    "keras": POC_Generation_KERAS_Config,
    "tensorflow": POC_Generation_TensorFlow_Config,
    "torch": POC_Generation_PyTorch_Config
}


def issue_dict_validator(issue_dict_str: str) -> str:
    try:
        issue_dict = json.loads(issue_dict_str)
        if not isinstance(issue_dict, dict):
            return "The output should be a JSON object (dict)."
        required_keys = {"score", "issue_title", "issue_description"}
        if not required_keys.issubset(issue_dict.keys()):
            return f"The JSON object must contain the keys: {required_keys}."
        if not isinstance(issue_dict["score"], int) or not (0 <= issue_dict["score"] <= 10):
            return "The 'score' value must be an integer between 0 and 10."
        if not isinstance(issue_dict["issue_title"], str):
            return "The 'issue_title' value must be a string."
        if not isinstance(issue_dict["issue_description"], str):
            return "The 'issue_description' value must be a string."
    except json.JSONDecodeError as e:
        return f"Invalid JSON format: {e}"
    return ""

Issue_Generation_TensorFlow_Config = PromptConfig(
    prompt_CN="""
    你是熟练掌握TensorFlow框架的一位深度学习工程师，在对TensorFlow-2.21.0框架中的一个API进行测试时发现了一个缺陷。
    复现缺陷的代码已经准备好了，能够成功复现该缺陷。现在你的任务是根据这个复现代码及代码运行结果，判断这个代码是否真正复现了一个bug。并生成一个issue描述，以便开发人员能够快速理解和修复这个问题。
    具体任务：
    1. 以0-10的分数评估这个代码是否真正复现了一个bug，10分表示完全复现了一个明确的bug，0分表示没有复现出任何问题。
    2. 生成评分的理由，要求简洁明了，能够清晰地说明你为什么给这个代码打这个分数。
    3. 生成一个issue标题，要求简洁明了，能够概括这个问题的核心。
    4. 生成一个issue描述，要求符合markdown格式，尽量简洁，能够让开发人员快速理解这个问题的背景、复现步骤、预期结果和实际结果。其中包含的复现代码应该尽量短，且能够成功运行并复现该缺陷。
    请你根据以下信息完成上述任务。请以
```json
{{
    "score": score,
    "reason": reason,
    "issue_title": issue_title,
    "issue_description": issue_description
}}
```
的形式返回你的答案，其中score是一个整数，reason,issue_title和issue_description都是字符串。
    API的名称：{api_name}。
    API的文档：{api_doc}。
    复现代码：{code_snippet}。
    复现代码的运行结果：{code_output}。
    """,
    prompt_EN="""
    You are a deep learning engineer proficient in the TensorFlow framework, and while testing an API in the TensorFlow-2.21.0 framework, you found a defect.
    The code to reproduce the defect is ready and can successfully reproduce the defect. Your current task is to determine whether this code truly reproduces a bug based on the reproduction code and its execution results, and generate an issue description so that developers can quickly understand and fix the problem.
    Specific tasks:
    1. Evaluate whether this code truly reproduces a bug on a scale of 0-10, where 10 means it fully reproduces a clear bug, and 0 means it does not reproduce any issue.
    2. Generate a reason for the score, which should be concise and clearly explain why you gave this score.
    3. Generate an issue title that is concise and can summarize the core of the problem.
    4. Generate an issue description that conforms to markdown format, is as concise as possible, and can allow developers to quickly understand the background of the problem, reproduction steps, expected results, and actual results. The reproduction code contained in it should be as short as possible and should be able to run successfully to reproduce the defect.
    Please complete the above tasks based on the following information. Please return your answer in the form of
```json
{{
    "score": score,
    "reason": reason,
    "issue_title": issue_title,
    "issue_description": issue_description
}}```
where score is an integer, reason, issue_title and issue_description are all strings.
    API name: {api_name}.
    API documentation: {api_doc}.
    Reproduction code: {code_snippet}.
    Execution results of the reproduction code: {code_output}.
    """,
    result_extractor=json_content_extractor,
    result_validator=issue_dict_validator,
    name = "Issue_Generation_TensorFlow_Config"
)


Issue_Generation_Pytorch_Config = PromptConfig(
    prompt_CN="""
    你是熟练掌握PyTorch框架的一位深度学习工程师，在对PyTorch-2.10.0+cu128框架中的一个API进行测试时发现了一个缺陷。
    复现缺陷的代码已经准备好了，能够成功复现该缺陷。现在你的任务是根据这个复现代码及代码运行结果，判断这个代码是否真正复现了一个bug。并生成一个issue描述，以便开发人员能够快速理解和修复这个问题。
    具体任务：
    1. 以0-10的分数评估这个代码是否真正复现了一个bug，10分表示完全复现了一个明确的bug，0分表示没有复现出任何问题。
    2. 生成评分的理由，要求简洁明了，能够清晰地说明你为什么给这个代码打这个分数。
    3. 生成一个issue标题，要求简洁明了，能够概括这个问题的核心。
    4. 生成一个issue描述，要求符合markdown格式，尽量简洁，能够让开发人员快速理解这个问题的背景、复现步骤、预期结果和实际结果。其中包含的复现代码应该尽量短，且能够成功运行并复现该缺陷。
    请你根据以下信息完成上述任务。请以
```json
{{
    "score": score,
    "reason": reason,
    "issue_title": issue_title,
    "issue_description": issue_description
}}
```
的形式返回你的答案，其中score是一个整数，reason,issue_title和issue_description都是字符串。
    API的名称：{api_name}。
    API的文档：{api_doc}。
    复现代码：{code_snippet}。
    复现代码的运行结果：{code_output}。
    """,
    prompt_EN="""
    You are a deep learning engineer proficient in the PyTorch framework, and while testing an API in the PyTorch-2.10.0+cu128 framework, you found a defect.
    The code to reproduce the defect is ready and can successfully reproduce the defect. Your current task is to determine whether this code truly reproduces a bug based on the reproduction code and its execution results, and generate an issue description so that developers can quickly understand and fix the problem.
    Specific tasks:
    1. Evaluate whether this code truly reproduces a bug on a scale of 0-10, where 10 means it fully reproduces a clear bug, and 0 means it does not reproduce any issue.
    2. Generate a reason for the score, which should be concise and clearly explain why you gave this score.
    3. Generate an issue title that is concise and can summarize the core of the problem.
    4. Generate an issue description that conforms to markdown format, is as concise as possible, and can allow developers to quickly understand the background of the problem, reproduction steps, expected results, and actual results. The reproduction code contained in it should be as short as possible and should be able to run successfully to reproduce the defect.
    Please complete the above tasks based on the following information. Please return your answer in the form of
```json
{{
    "score": score,
    "reason": reason,
    "issue_title": issue_title,
    "issue_description": issue_description
}}```
where score is an integer, reason, issue_title and issue_description are all strings.
    API name: {api_name}.
    API documentation: {api_doc}.
    Reproduction code: {code_snippet}.
    Execution results of the reproduction code: {code_output}.
    """,
    result_extractor=json_content_extractor,
    result_validator=issue_dict_validator,
    name = "Issue_Generation_Pytorch_Config"
)


Issue_Generation_Keras_Config = PromptConfig(
    prompt_CN="""
    你是熟练掌握Keras框架的一位深度学习工程师，在对Keras-3.13.2框架中的一个API进行测试时发现了一个缺陷。
    你用的Keras后端是TensorFlow-2.21.0。
    复现缺陷的代码已经准备好了，能够成功复现该缺陷。现在你的任务是根据这个复现代码及代码运行结果，判断这个代码是否真正复现了一个bug。并生成一个issue描述，以便开发人员能够快速理解和修复这个问题。
    具体任务：
    1. 以0-10的分数评估这个代码是否真正复现了一个bug，10分表示完全复现了一个明确的bug，0分表示没有复现出任何问题。
    2. 生成评分的理由，要求简洁明了，能够清晰地说明你为什么给这个代码打这个分数。
    3. 生成一个issue标题，要求简洁明了，能够概括这个问题的核心。
    4. 生成一个issue描述，要求符合markdown格式，尽量简洁，能够让开发人员快速理解这个问题的背景、复现步骤、预期结果和实际结果。其中包含的复现代码应该尽量短，且能够成功运行并复现该缺陷。
    请你根据以下信息完成上述任务。请以
```json
{{
    "score": score,
    "reason": reason,
    "issue_title": issue_title,
    "issue_description": issue_description
}}
    "issue_title": issue_title,
    "issue_description": issue_description
}}
```
的形式返回你的答案，其中score是一个整数，reason,issue_title和issue_description都是字符串。
    API的名称：{api_name}。
    API的文档：{api_doc}。
    复现代码：{code_snippet}。
    复现代码的运行结果：{code_output}。
    """,
    prompt_EN="""
    You are a deep learning engineer proficient in the Keras framework, and while testing an API in the Keras-3.13.2 framework, you found a defect.
    Your Keras backend is TensorFlow-2.21.0.
    The code to reproduce the defect is ready and can successfully reproduce the defect. Your current task is to determine whether this code truly reproduces a bug based on the reproduction code and its execution results, and generate an issue description so that developers can quickly understand and fix the problem.
    Specific tasks:
    1. Evaluate whether this code truly reproduces a bug on a scale of 0-10, where 10 means it fully reproduces a clear bug, and 0 means it does not reproduce any issue.
    2. Generate a reason for the score, which should be concise and clearly explain why you gave this score.
    3. Generate an issue title that is concise and can summarize the core of the problem.
    4. Generate an issue description that conforms to markdown format, is as concise as possible, and can allow developers to quickly understand the background of the problem, reproduction steps, expected results, and actual results. The reproduction code contained in it should be as short as possible and should be able to run successfully to reproduce the defect.
    Please complete the above tasks based on the following information. Please return your answer in the form of
```json{{
    "score": score,
    "reason": reason,
    "issue_title": issue_title,
    "issue_description": issue_description
}}```
where score is an integer, reason, issue_title and issue_description are all strings.
    API name: {api_name}.
    API documentation: {api_doc}.
    Reproduction code: {code_snippet}.
    Execution results of the reproduction code: {code_output}.
    """,
    result_extractor=json_content_extractor,
    result_validator=issue_dict_validator,
    name = "Issue_Generation_Keras_Config"
)

issue_generation_config_dict = {
    "keras": Issue_Generation_Keras_Config,
    "tensorflow": Issue_Generation_TensorFlow_Config,
    "torch": Issue_Generation_Pytorch_Config
}