<script lang="ts">
  export let inStock: boolean;
</script>

{#if inStock}
  <span class="badge badge-in">IN</span>
{:else}
  <span class="badge badge-out">OUT</span>
{/if}

<style>
  .badge { display: inline-block; padding: 0.1rem 0.45rem;
    border-radius: 3px; font-size: 0.7rem; font-weight: 700;
    min-width: 2rem; text-align: center; }
  .badge-in { background: #d4edda; color: #155724; }
  .badge-out { background: #f8d7da; color: #721c24; }
</style>
