import { component$, useSignal } from "@builder.io/qwik";
import type { DocumentHead } from "@builder.io/qwik-city";

export default component$(() => {
  const count = useSignal(0);

  return (
    <main>
      <h1>Hello World</h1>
      <div>
        <p>Count: {count.value}</p>
        <button onClick$={() => count.value++}>Increment</button>
      </div>
    </main>
  );
});

export const head: DocumentHead = {
  title: "Hello World",
  meta: [
    {
      name: "description",
      content: "Hello World page",
    },
  ],
};
