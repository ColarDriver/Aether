export class CircularBuffer<T> {
  readonly capacity: number
  #items: T[] = []
  #start = 0
  #length = 0

  constructor(capacity: number) {
    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new RangeError('CircularBuffer capacity must be a positive integer')
    }
    this.capacity = capacity
    this.#items = new Array<T>(capacity)
  }

  get size(): number {
    return this.#length
  }

  push(item: T): void {
    const index = (this.#start + this.#length) % this.capacity
    this.#items[index] = item
    if (this.#length === this.capacity) {
      this.#start = (this.#start + 1) % this.capacity
      return
    }
    this.#length += 1
  }

  clear(): void {
    this.#items = new Array<T>(this.capacity)
    this.#start = 0
    this.#length = 0
  }

  toArray(): T[] {
    const out: T[] = []
    for (let i = 0; i < this.#length; i += 1) {
      out.push(this.#items[(this.#start + i) % this.capacity] as T)
    }
    return out
  }
}
