using PkgTemplates

t = Template(;
            user = "gionikola",
            license = "MIT", 
            dir     = "package", 
            authors = ["Giorgi Nikolaishvili"],
            plugins = [
            ],
)

generate("DynamicFactorModeling", t)