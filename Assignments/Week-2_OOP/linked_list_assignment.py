# Assignment 2 - Implement a Linked List in Python Using OOP and Delete the Nth Node

class Node:
    """
    A class to represent a single node in the linked list.
    
    Attributes:
        data: The value stored in the node
        next: Reference to the next node in the list
    """
    
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    """
    A class to represent a singly linked list with various operations.
    
    Attributes:
        head: Reference to the first node in the list
    """
    
    def __init__(self):
        self.head = None
    
    def add_node(self, data):
        """Add a node to the end of the list."""
        new_node = Node(data)
        
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
    
    def print_list(self):
        """Print the list."""
        if not self.head:
            print("List is empty")
            return
        
        current = self.head
        while current:
            print(current.data, end=" -> " if current.next else " -> None\n")
            current = current.next
    
    def delete_nth_node(self, n):
        """Delete the nth node (1-based index)."""
        if not self.head:
            raise ValueError("Cannot delete from an empty list")
        
        if n < 1:
            raise IndexError("Index must be 1 or greater")
        
        # Delete first node
        if n == 1:
            self.head = self.head.next
            return
        
        # Find the node before the one to delete
        current = self.head
        for i in range(1, n - 1):
            if not current.next:
                raise IndexError("Index out of range")
            current = current.next
        
        # Check if nth node exists
        if not current.next:
            raise IndexError("Index out of range")
        
        # Delete the nth node
        current.next = current.next.next


# Test the implementation
if __name__ == "__main__":
    # Create a linked list
    ll = LinkedList()
    
    # Add some nodes
    ll.add_node(10)
    ll.add_node(20)
    ll.add_node(30)
    ll.add_node(40)
    ll.add_node(50)
    
    print("Original list:")
    ll.print_list()
    
    # Delete 3rd node
    ll.delete_nth_node(3)
    print("\nAfter deleting 3rd node:")
    ll.print_list()
    
    # Test edge cases
    try:
        ll.delete_nth_node(10)  # Out of range
    except IndexError as e:
        print(f"\nError: {e}")
    
    # Test deleting from empty list
    empty_list = LinkedList()
    try:
        empty_list.delete_nth_node(1)
    except ValueError as e:
        print(f"Error: {e}")