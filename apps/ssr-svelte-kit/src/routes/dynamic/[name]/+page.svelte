<script>
  import { page } from '$app/stores';
  let count = 0;
  $: name = $page.params.name;
</script>

<main>
  <h1>Hello World</h1>
  <p>Dynamic ID: {name}</p>
  <div>
    <p>Count: {count}</p>
    <button on:click={() => count++}>Increment</button>
  </div>
</main>
