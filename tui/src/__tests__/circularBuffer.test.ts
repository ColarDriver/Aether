import { describe, expect, it } from 'vitest'

import { CircularBuffer } from '../lib/circularBuffer.js'

describe('CircularBuffer', () => {
  it('keeps items in insertion order', () => {
    const buffer = new CircularBuffer<number>(3)

    buffer.push(1)
    buffer.push(2)

    expect(buffer.size).toBe(2)
    expect(buffer.toArray()).toEqual([1, 2])
  })

  it('drops the oldest item when capacity is exceeded', () => {
    const buffer = new CircularBuffer<number>(3)

    buffer.push(1)
    buffer.push(2)
    buffer.push(3)
    buffer.push(4)

    expect(buffer.size).toBe(3)
    expect(buffer.toArray()).toEqual([2, 3, 4])
  })

  it('clears buffered items without changing capacity', () => {
    const buffer = new CircularBuffer<string>(2)

    buffer.push('a')
    buffer.clear()
    buffer.push('b')

    expect(buffer.capacity).toBe(2)
    expect(buffer.size).toBe(1)
    expect(buffer.toArray()).toEqual(['b'])
  })

  it('rejects non-positive capacities', () => {
    expect(() => new CircularBuffer(0)).toThrow(RangeError)
    expect(() => new CircularBuffer(1.5)).toThrow(RangeError)
  })
})
