using PkgTemplates

t = Template(;
            user = "gionikola",
            license = "MIT", 
            authors = ["Giorgi Nikolaishvili"],
            plugins = [
                TravisCI(),
                Codecov(),
                Coveralls(),
                AppVeyor()
            ],
)

generate("DynamicFactorModeling", t)