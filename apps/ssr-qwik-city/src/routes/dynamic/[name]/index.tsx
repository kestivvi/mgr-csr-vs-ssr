import { component$ } from "@builder.io/qwik";
import { useLocation } from "@builder.io/qwik-city";
import type { DocumentHead } from "@builder.io/qwik-city";

export default component$(() => {
  const location = useLocation();
  const name = location.params.name || "Guest";

  return <h1>Hello, {name}</h1>;
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
