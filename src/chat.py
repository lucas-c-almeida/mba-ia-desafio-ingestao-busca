from search import search_prompt


def main():
    print("Chat baseado no conteúdo do PDF. Digite 'sair' para encerrar.\n")
    while True:
        try:
            pergunta = input("Faça sua pergunta: ").strip()
            if not pergunta:
                continue
            if pergunta.lower() in ("sair", "exit", "quit"):
                print("Encerrando.")
                break
            resposta = search_prompt(pergunta)
            print(f"\nPERGUNTA: {pergunta}")
            print(f"RESPOSTA: {resposta}\n")
        except KeyboardInterrupt:
            print("\nEncerrando.")
            break


if __name__ == "__main__":
    main()
