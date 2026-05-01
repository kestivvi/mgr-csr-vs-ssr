import { component$ } from "@builder.io/qwik";
import { useLocation } from "@builder.io/qwik-city";
import type { DocumentHead } from "@builder.io/qwik-city";

export default component$(() => {
  const location = useLocation();
  const name = location.params.name || "Guest";

  return (
    <div>
      <h1>Hello World</h1>
      <p>Dynamic ID: {name}</p>
    </div>
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
