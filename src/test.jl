using PkgTemplates

t = Template(;
            user = "gionikola",
            license = "MIT", 
            dir     = "package", 
            authors = ["Giorgi Nikolaishvili"],
            plugins = [
                TravisCI(),
                Codecov(),
                Coveralls(),
                AppVeyor()
            ],
)

generate("DynamicFactorModeling", t)