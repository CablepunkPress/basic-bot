declare const marked: {
  parse(text: string): string;
  Renderer: {
    new(): {
      link: (args: object) => string;
    };
  };
  setOptions(options: object): void;
};
