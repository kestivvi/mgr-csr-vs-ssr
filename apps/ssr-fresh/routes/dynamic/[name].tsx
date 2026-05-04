import { PageProps } from "fresh";
import Counter from "../../islands/Counter.tsx";

export default function DynamicPage(props: PageProps) {
  const { name } = props.params;

  return (
    <main>
      <h1>Hello World</h1>
      <p>Dynamic ID: {name}</p>
      <Counter />
    </main>
  );
}
