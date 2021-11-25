using PkgTemplates

t = Template(;
            user = "gionikola",
            license = "MIT", 
            authors = ["Giorgi Nikolaishvili"],
            plugins = [
            ],
)

generate("DynamicFactorModeling", t)