NAME:=pydata_mlflow
OUT:=$(NAME).html $(NAME).ipynb
IN:=*_files images/*
DEPS=style.css

html: $(NAME).html
all: html
zip: $(NAME).zip

%.html: %.qmd $(DEPS)
	quarto render $< --to revealjs -o $@

%.ipynb: %.qmd $(DEPS)
	quarto convert $<

%.zip: $(OUT) $(IN) $(DEPS)
	zip -r $(NAME).zip $(OUT) $(IN)

clean:
	rm $(NAME).zip $(NAME).html $(NAME).ipynb
