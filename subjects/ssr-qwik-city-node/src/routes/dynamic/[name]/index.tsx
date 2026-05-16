import { component$, useSignal } from "@builder.io/qwik";
import { useLocation, type DocumentHead } from "@builder.io/qwik-city";

export default component$(() => {
  const loc = useLocation();
  const name = loc.params.name;
  const count = useSignal(0);

  return (
    <main>
      <h1>Hello World</h1>
      <p>Dynamic ID: {name}</p>
      <div>
        <p>Count: {count.value}</p>
        <button onClick$={() => count.value++}>Increment</button>
      </div>
    </main>
  );
});

export const head: DocumentHead = {
  title: "Dynamic Page",
  meta: [
    {
      name: "description",
      content: "Dynamic page with parameter",
    },
  ],
};
