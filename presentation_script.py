# To run this script, just run the following from the root of the repository:
# python3 presentation_script.py

# Slides 18-19:
with open("testing/BaselinePortfolioTest.py") as f:
    code = compile(f.read(), "main_script.py", 'exec')
    exec(code)

# Slides 20-21: 
with open("simulator.py") as f:
    code = compile(f.read(), "main_script.py", 'exec')
    exec(code)


# Slide 22:
with open("testing/Test1.py") as f:
    code = compile(f.read(), "main_script.py", 'exec')
    exec(code)



