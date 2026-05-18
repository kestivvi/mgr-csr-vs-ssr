<script lang="ts">
  export let inStock: boolean;
</script>

{#if inStock}
  <span class="badge badge-in">IN</span>
{:else}
  <span class="badge badge-out">OUT</span>
{/if}
