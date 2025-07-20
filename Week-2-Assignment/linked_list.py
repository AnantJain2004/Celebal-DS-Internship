# Assignment 2 - Implement a Linked List in Python Using OOP and Delete the Nth Node

class Node:
    """
    A class to represent a single node in the linked list.
    
    Attributes:
        data: The value stored in the node
        next: Points to the next node in the list
    """
    
    def __init__(self, data):
        """
        Initialize a new node with given data.
        
        Args:
            data: The value to store in the node
        """
        self.data = data
        self.next = None
    
    def __str__(self):
        """String representation of the node."""
        return str(self.data)


class LinkedList:
    """
    A class to represent a singly linked list with various operations.
    
    Attributes:
        head: Points to the first node in the list
        size: Current number of nodes in the list
    """
    
    def __init__(self):
        """Initialize an empty linked list."""
        self.head = None
        self.size = 0
    
    def is_empty(self):
        """
        Check if the linked list is empty.
        
        Returns:
            bool: True if list is empty, False otherwise
        """
        return self.head is None
    
    def get_size(self):
        """
        Get the current size of the linked list.
        
        Returns:
            int: Number of nodes in the list
        """
        return self.size
    
    def add_node(self, data):
        """
        Add a new node to the end of the linked list.
        
        Args:
            data: The value to add to the node
        """
        new_node = Node(data)
        
        # If list is empty, make new node the head
        if self.is_empty():
            self.head = new_node
        else:
            # Traverse to the end of the list
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        
        self.size += 1
        print(f"Added node with data: {data}")
    
    def print_list(self):
        """
        Print all elements in the linked list.
        """
        if self.is_empty():
            print("The list is empty!")
            return
        
        current = self.head
        elements = []
        
        while current:
            elements.append(str(current.data))
            current = current.next
        
        print("Linked List: " + " -> ".join(elements) + " -> None")
    
    def delete_nth_node(self, n):
        """
        Delete the nth node from the linked list (1-based indexing).
        
        Args:
            n (int): The position of the node to delete (1-based)
        
        Returns:
            IndexError: If the index is out of range
            ValueError: If the list is empty
        """
        try:
            # Check if list is empty
            if self.is_empty():
                raise ValueError("Cannot delete from an empty list!")
            
            # Check if index is valid
            if n < 1 or n > self.size:
                raise IndexError(f"Index {n} is out of range. List has {self.size} elements.")
            
            # Special case: delete the first node
            if n == 1:
                deleted_data = self.head.data
                self.head = self.head.next
                self.size -= 1
                print(f"Deleted node at position {n} with data: {deleted_data}")
                return deleted_data
            
            # Find the node before the one to delete
            current = self.head
            for i in range(1, n - 1):
                current = current.next
            
            # Store the data of the node to be deleted
            deleted_data = current.next.data
            
            # Remove the node by updating the link
            current.next = current.next.next
            self.size -= 1
            
            print(f"Deleted node at position {n} with data: {deleted_data}")
            return deleted_data
            
        except ValueError as e:
            print(f"Error: {e}")
            raise
        except IndexError as e:
            print(f"Error: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
            raise
    

def implement_linked_list():
    """
    Demonstrate the linked list implementation with various operations.
    """
    print("=" * 60)
    print("LINKED LIST IMPLEMENTATION")
    print("=" * 60)
    
    # Create a new linked list
    ll = LinkedList()
    
    print("\n1. Creating an empty linked list:")
    print(f"Is empty: {ll.is_empty()}")
    print(f"Size: {ll.get_size()}")
    ll.print_list()
    
    print("\n2. Adding nodes to the list:")
    # Add some nodes
    sample_data = [10, 20, 30, 40, 50]
    for data in sample_data:
        ll.add_node(data)
    
    print(f"\nAfter adding nodes - Size: {ll.get_size()}")
    ll.print_list()
    
    print("\n3. Testing delete operations:")
    
    # Test deleting from different positions
    delete_labels = ["first", "middle", "last"]
    for label in delete_labels:
        if ll.get_size() > 0:
            print(f"\nBefore deletion - Size: {ll.get_size()}")
            ll.print_list()

            if label == "first": # first position
                pos = 1
            elif label == "middle": # middle position
                pos = (ll.get_size() + 1) // 2
            else:  # last position
                pos = ll.get_size()

            try:
                ll.delete_nth_node(pos)
                print(f"After deletion - Size: {ll.get_size()}")
                ll.print_list()
            except (IndexError, ValueError) as e:
                print(f"Deletion failed: {e}")
    
    print("\n4. Testing edge cases:")
    
    # Test deleting from empty list
    print("\nTesting deletion from empty list:")
    ll = LinkedList()  # Create empty list
    try:
        ll.delete_nth_node(1)
    except ValueError as e:
        print(f"Expected error caught: {e}")
    
    # Test deleting with invalid index
    print("\nTesting deletion with invalid index:")
    ll.add_node(100)
    ll.print_list()
    try:
        ll.delete_nth_node(5)  # Index out of range
    except IndexError as e:
        print(f"Expected error caught: {e}")
    
    try:
        ll.delete_nth_node(0)  # Invalid index (less than 1)
    except IndexError as e:
        print(f"Expected error caught: {e}")
    
    print("\n5. Final state of the linked list:")
    ll.print_list()
    print(f"Final size: {ll.get_size()}")
    
    print("\n" + "=" * 60)
    print("IMPLEMENTATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    # Run the implementation
    implement_linked_list()