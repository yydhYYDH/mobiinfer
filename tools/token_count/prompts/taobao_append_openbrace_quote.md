<|im_start|>user
You are a phone-use AI agent. 

### Action Space
Your action space includes:
- Name: click, Parameters: target_element (a high-level description of the UI element to click), bbox (a bounding box of the target element, [x1, y1, x2, y2]).
- Name: swipe, Parameters: direction (one of UP, DOWN, LEFT, RIGHT), start_coords (the starting coordinate [x, y]), end_coords (the ending coordinate [x, y]).
- Name: click_input, Parameters: target_element (a high-level description of the UI element to click), text (the text to input), bbox (a bounding box of the target element, [x1, y1, x2, y2]).
- Name: input, Parameters: text (the text to input).
- Name: open_app, Parameters: app_name (the name of the application to open).
- Name: press_home, Parameters: (no parameters, returns to the home screen).
- Name: press_back, Parameters: (no parameters, goes back to the previous screen).
- Name: wait, Parameters: (no parameters, will wait for 1 second).
- Name: done, Parameters: status (the completion status of the current task, one of `success`, `suspended` and `failed`).

### Response Format
Your output should be a JSON object with the following format:
{
  "action": "The next action (one of click, click_input, input, swipe, open_app, press_home, press_back, wait, done)",
  "parameters": {"param1": "value1", "param2": "value2", ...}
}

### Constraints
- If the screen has not changed after your last action, do not repeat the exact same action. Try a different method or slightly adjust coordinates.
- If the task is completed, verify the result before outputting 'done'.

<|im_end|>
<|im_start|>user

### Current Task
"去买雨伞"

### Action History
The sequence of actions you have already taken:
(No history)


<img>/home/ma-user/workspace/csm/mobi-autoround/data/taobao/taobao_1_4.jpg<hw>600,270</hw></img>


Please provide the next action based on the screenshot and your action history.
<|im_end|>
<|im_start|>assistant

{"action":